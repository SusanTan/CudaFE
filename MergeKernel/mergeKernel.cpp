#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/CFG.h"


#include <set>
#include <stack>
#include <map>
#include "IDMap.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/Operator.h"

using namespace llvm;

namespace {
class KernelProfile{
  public:
  std::vector<Value*> loopDims;
  std::map<Value*, int> dim2classify; //0: no meaning 1:outermost loop of grid 2:outermost loop of block
  int gridLoopCnt = 0;
  int blockLoopCnt = 0;
  BasicBlock *kernelBB = nullptr;
  Function *deviceKernel = nullptr;
  CallInst *kernelCall = nullptr;
  Function *newFunc = nullptr;
};

struct MergeKernel : public ModulePass {
  static char ID;
  std::set<Function*>funcs2delete;
  std::map<Function*, Function*> device2newFunc;
  int mdID = 0;
  MergeKernel() : ModulePass(ID) {}

  void findThreadDim(KernelProfile *kernelProfile, Function &F, LoadInst *DimArg, bool isBlockDim){
    //LoadInst *DimArg = dyn_cast<LoadInst>(CI->getArgOperand(2));
    assert(DimArg && "mergeKernel: DimArg is not a load inst\n");
    GetElementPtrInst *ArgGep = dyn_cast<GetElementPtrInst>(DimArg->getOperand(0));
    assert(ArgGep && "mergeKernel: DimGep is not a gep inst\n");
    AllocaInst *Coerce = dyn_cast<AllocaInst>(ArgGep->getOperand(0));
    assert(Coerce && "mergeKernel: DimAllocaCoerce is not an alloca inst\n");
    BitCastInst *CoerceBitCast = nullptr;
    for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
      BitCastInst *bitcast = dyn_cast<BitCastInst>(&*I);
      if(!bitcast) continue;
      if(bitcast->getOperand(0) != Coerce) continue;
      CoerceBitCast = bitcast;
      break;
    }
    assert(CoerceBitCast && "mergeKernel: CoerceBitCast is not a bit cast inst\n");
    BitCastInst *AggBitCast = nullptr;
    for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
      CallInst *CI = dyn_cast<CallInst>(&*I);
      if(!CI) continue;
      if(CI->getCalledFunction()->getName() != "llvm.memcpy.p0i8.p0i8.i64") continue;
      if(CI->getArgOperand(0) != CoerceBitCast) continue;
      BitCastInst *bitcast = dyn_cast<BitCastInst>(CI->getArgOperand(1));
      AggBitCast = bitcast;
      break;
    }
    assert(AggBitCast && "mergeKernel: AggMemcpy not found \n");
    AllocaInst *AggAlloca = dyn_cast<AllocaInst>(AggBitCast->getOperand(0));
    assert(AggAlloca && "mergeKernel: gridDimAlloca is not an alloca inst \n");
    bool foundDim = false;
    for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
      CallInst *CI = dyn_cast<CallInst>(&*I);
      if(!CI) continue;
      if(!CI->getCalledFunction()->getName().contains("_ZN4dim3C2Ejjj")) continue;
      if(CI->getArgOperand(0) != AggAlloca) continue;

      for(int i=1; i<=3; ++i){
        Value *dim = CI->getArgOperand(i);
        //if(ConstantInt *dimConst = dyn_cast<ConstantInt>(dim))
        //  if(dimConst->getZExtValue() == 1){
        //    continue;
        //  }
        kernelProfile->loopDims.push_back(dim);
        if(isBlockDim){
          bool isOne = false;
          if(ConstantInt *constInt = dyn_cast<ConstantInt>(dim))
            if(constInt->getSExtValue() == 1)
              isOne = true;
          if(!isOne) kernelProfile->blockLoopCnt ++;
          if(i==1)
            kernelProfile->dim2classify[dim] = 2;
          else
            kernelProfile->dim2classify[dim] = 0;
        }
        else{
          bool isOne = false;
          if(ConstantInt *constInt = dyn_cast<ConstantInt>(dim))
            if(constInt->getSExtValue() == 1)
              isOne = true;
          if(!isOne) kernelProfile->gridLoopCnt ++;
          if(i==1)
            kernelProfile->dim2classify[dim] = 1;
          else
            kernelProfile->dim2classify[dim] = 0;
        }

        errs() << "mergeKernel: Dim " << i << " : " << *dim <<"\n";
      }


      foundDim = true;
      break;
    }
    if(!foundDim){
      BitCastInst *AggBitCast2 = nullptr;
      for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
        BitCastInst *bitcast = dyn_cast<BitCastInst>(&*I);
        if(!bitcast) continue;
        if(bitcast->getOperand(0) != AggAlloca) continue;
        AggBitCast2 = bitcast;
        break;
      }
      assert(AggBitCast2 && "mergeKernel: AggBitCast2 is not found \n");
      BitCastInst *DimBitCast = nullptr;
      for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
        CallInst *CI = dyn_cast<CallInst>(&*I);
        if(!CI) continue;
        if(CI->getCalledFunction()->getName() != "llvm.memcpy.p0i8.p0i8.i64") continue;
        if(CI->getArgOperand(0) != AggBitCast2) continue;
        BitCastInst *bitcast = dyn_cast<BitCastInst>(CI->getArgOperand(1));
        DimBitCast = bitcast;
        break;
      }
      AllocaInst *DimAlloca = dyn_cast<AllocaInst>(DimBitCast->getOperand(0));
      assert(DimAlloca && "mergeKernel: DimAlloca is not found \n");
      for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
        CallInst *CI = dyn_cast<CallInst>(&*I);
        if(!CI) continue;
        if(!CI->getCalledFunction()->getName().contains("_ZN4dim3C2Ejjj")) continue;
        if(CI->getArgOperand(0) != DimAlloca) continue;
        for(int i=1; i<=3; ++i){
          Value *dim = CI->getArgOperand(i);
          //if(ConstantInt *dimConst = dyn_cast<ConstantInt>(dim))
          //  if(dimConst->getZExtValue() == 1)
          //    continue;
          kernelProfile->loopDims.push_back(dim);
          if(isBlockDim){
            bool isOne = false;
            if(ConstantInt *constInt = dyn_cast<ConstantInt>(dim))
              if(constInt->getSExtValue() == 1)
                isOne = true;
            if(!isOne) kernelProfile->blockLoopCnt ++;
            if(i==1)
              kernelProfile->dim2classify[dim] = 2;
            else
              kernelProfile->dim2classify[dim] = 0;
          }
          else{
            bool isOne = false;
            if(ConstantInt *constInt = dyn_cast<ConstantInt>(dim))
              if(constInt->getSExtValue() == 1)
                isOne = true;
            if(!isOne) kernelProfile->gridLoopCnt ++;
            if(i==1)
              kernelProfile->dim2classify[dim] = 1;
            else
              kernelProfile->dim2classify[dim] = 0;
          }
        }
        foundDim = true;
        break;
      }
    }
  }

  bool runOnModule(Module &M) override {
    //transform target
    for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
      std::map<CallInst*, KernelProfile*> kernelProfiles;
      Function *F = &*FI;
      F->removeFnAttr("target-features");
      F->removeFnAttr("target-cpu");
      BasicBlock *kernelBB = nullptr;
      std::vector<Instruction*> insts2Remove;
      for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
        if(CallInst *CI = dyn_cast<CallInst>(&*I)){
          Function* calledFunc = CI->getCalledFunction();
          if(calledFunc->getName().contains("llvm.nvvm.barrier")){
            funcs2delete.insert(calledFunc);
            insts2Remove.push_back(CI);
          }
          else if(calledFunc->getName().contains("_ZN4dim3C2Ejjj")){
            funcs2delete.insert(calledFunc);
            insts2Remove.push_back(CI);

            Value *dimPtr = CI->getArgOperand(0);
            PointerType *dimPtrTy = dyn_cast<PointerType>(dimPtr->getType());
            assert(dimPtrTy && "dimPtr isn't of pointer type!\n");
            auto i32Ty = Type::getInt32Ty(CI->getContext());
            for(int i=1; i<=3; ++i){
              Value *dim = CI->getArgOperand(i);
              //replace _ZN4dim3C2Ejjj with stores to dim3 struct
              std::vector<Value*>idxList;
              idxList.push_back( ConstantInt::get(i32Ty, 0));
              idxList.push_back( ConstantInt::get(i32Ty, i-1));
              auto gep = GetElementPtrInst::Create(dimPtrTy->getPointerElementType(), dimPtr, idxList, "dim3gep."+std::to_string(i-1), CI);
              //StoreInst (Value *Val, Value *Ptr, Instruction *InsertBefore)
              new StoreInst(dim, gep, CI);
            }
        }
        else if(calledFunc->getName().contains("cudaFree")){
           // auto ptr = CI->getArgOperand(0);
           // CallInst::CreateFree(ptr, CI);
            insts2Remove.push_back(CI);
        }
        else if(calledFunc->getName().contains("cudaConfigureCall")){
            //temporous registers
            CallInst *kernelCall = nullptr;
            Function *deviceKernel = nullptr;
            Function *newFunc = nullptr;

            errs() << "CudaFE: found cudaConfigureCall\n";
            kernelProfiles[CI] = new KernelProfile();
            //remove control flow caused by configuration failure
            BranchInst *CF2Remove = dyn_cast<BranchInst>(CI->getParent()->getTerminator());
            CmpInst *cmp = dyn_cast<CmpInst>(CF2Remove->getCondition());
            if(cmp){
              if(cmp->getPredicate() == CmpInst::ICMP_EQ || cmp->getPredicate() == CmpInst::ICMP_NE){
                auto opnd0 = cmp->getOperand(0);
                auto opnd1 = cmp->getOperand(1);
                bool ConfigureThenBranchPattern = false;
                if(ConstantInt *integer = dyn_cast<ConstantInt>(opnd0)){
                  if(integer->getZExtValue() == 0 && opnd1 == CI && cmp->getPredicate() == CmpInst::ICMP_EQ)
                    ConfigureThenBranchPattern = false;
                  else if (integer->getZExtValue() == 0 && opnd1 == CI && cmp->getPredicate() == CmpInst::ICMP_NE)
                    ConfigureThenBranchPattern = true;
                } else if(ConstantInt *integer = dyn_cast<ConstantInt>(opnd1)){
                  if(integer->getZExtValue() == 0 && opnd0 == CI && cmp->getPredicate() == CmpInst::ICMP_EQ)
                    ConfigureThenBranchPattern = true;
                  else if (integer->getZExtValue() == 0 && opnd0 == CI && cmp->getPredicate() == CmpInst::ICMP_NE)
                    ConfigureThenBranchPattern = false;
                }
                if(ConfigureThenBranchPattern){
                  CF2Remove->setCondition(ConstantInt::get(Type::getInt1Ty(CF2Remove->getContext()), 1));
                  kernelProfiles[CI]->kernelBB = CF2Remove->getSuccessor(0);
                  errs() << "mergeKernel: kernelBB: " << *(kernelProfiles[CI]->kernelBB) << "\n";
                } else {
                  CF2Remove->setCondition(ConstantInt::get(Type::getInt1Ty(CF2Remove->getContext()), 0));
                  kernelProfiles[CI]->kernelBB = CF2Remove->getSuccessor(1);
                  errs() << "mergeKernel: kernelBB: " << *(kernelProfiles[CI]->kernelBB) << "\n";
                }

                //write metadata to the function call
                for(auto &I : *(kernelProfiles[CI]->kernelBB)){
                  if(CallInst *ci = dyn_cast<CallInst>(&I)){
                    Function *calledFunc = ci->getCalledFunction();
                    for (inst_iterator I = inst_begin(calledFunc), E = inst_end(calledFunc); I != E; ++I) {
                      if(CallInst *callInst = dyn_cast<CallInst>(&*I)){
                        Function *calledF = callInst->getCalledFunction();
                        if(calledF->getName().contains("cudaLaunch")){
                          kernelProfiles[CI]->kernelCall = ci;
                          kernelCall = ci;
                          break;
                        }
                      }
                    }
                  }

                  if(kernelCall)
                    break;
                }

                assert(kernelCall && "mergeKernel: din't find kernel call!\n");
                errs() << "mergeKernel: kernel call: " << *(kernelCall) << "\n";
                insts2Remove.push_back(cmp);
            }
          }

            assert(kernelCall && "mergeKernel: din't find kernel call!\n");
            errs() << "mergeKernel: kernel call: " << *(kernelCall) << "\n";

            //TODO: mem2reg sort of changes everything with the pattern matching
            errs() << "MergeKernel: blocks per grid:\n";
            if(isa<ConstantInt>(CI->getArgOperand(0))){
              Value *dim = CI->getArgOperand(0);
              kernelProfiles[CI]->loopDims.push_back(dim);
              kernelProfiles[CI]->dim2classify[dim] = 1;
              bool isOne = false;
              if(ConstantInt *constInt = dyn_cast<ConstantInt>(dim))
                if(constInt->getSExtValue() == 1)
                  isOne = true;
              if(!isOne) kernelProfiles[CI]->gridLoopCnt ++;
            }
            else{
              findThreadDim(kernelProfiles[CI], *F, dyn_cast<LoadInst>(CI->getArgOperand(0)), false);
            }

            errs() << "MergeKernel: threads per block:\n";
            if(isa<ConstantInt>(CI->getArgOperand(2))){
              Value *dim = CI->getArgOperand(2);
              kernelProfiles[CI]->loopDims.push_back(dim);
              kernelProfiles[CI]->dim2classify[dim] = 2;
              bool isOne = false;
              if(ConstantInt *constInt = dyn_cast<ConstantInt>(dim))
                if(constInt->getSExtValue() == 1)
                  isOne = true;
              if(!isOne) kernelProfiles[CI]->blockLoopCnt ++;
            }
            else{
              findThreadDim(kernelProfiles[CI], *F, dyn_cast<LoadInst>(CI->getArgOperand(2)), true);
            }
            //find host kernel function and actual kernel name
            Module *M = F->getParent();
            auto hostKernel = kernelCall->getCalledFunction();
            funcs2delete.insert(hostKernel);
            StringRef host_kernelName = hostKernel->getName();
            auto namePair = host_kernelName.rsplit("_CudaFE_");
            StringRef kernelName = namePair.second;
            if(kernelName == "") kernelName = host_kernelName;
            errs() << "mergeKernel: found kernelName: " << kernelName << "\n";

            //Find device kernel function
            for (Module::iterator FI = M->begin(), FE = M->end(); FI != FE; ++FI) {
              Function *F = &*FI;
              auto funcName = F->getName();
              if(funcName.contains(kernelName) &&
                 funcName != host_kernelName ){
                kernelProfiles[CI]->deviceKernel = F;
                deviceKernel = F;
                funcs2delete.insert(deviceKernel);
                break;
              }
            }
            errs() << "MergeKernel: found deviceKernel: " << *deviceKernel << "\n";
            if(device2newFunc.find(deviceKernel) != device2newFunc.end()){
              auto newFunc = device2newFunc[deviceKernel];
              kernelProfiles[CI]->newFunc = newFunc;
              insts2Remove.push_back(CI);
              continue;
            }


            //create a new function for kernel
            Type* i32Ty = Type::getInt32Ty(deviceKernel->getContext());
            Type* i64Ty = Type::getInt64Ty(deviceKernel->getContext());
            std::vector<Type*>argTys;
            for(auto arg = deviceKernel->arg_begin(); arg != deviceKernel->arg_end(); ++arg)
              argTys.push_back(arg->getType());
            for(int i=0; i<12; i++)
              argTys.push_back(i32Ty);
            FunctionType* funcTy = FunctionType::get(
                  deviceKernel->getReturnType(), //return type
                  ArrayRef<Type*>(argTys), //arg types;
                  false
                );
            kernelProfiles[CI]->newFunc = Function::Create(
                  funcTy,
                  deviceKernel->getLinkage(),
                  deviceKernel->getName(),
                  deviceKernel->getParent()
                );
            newFunc = kernelProfiles[CI]->newFunc;
            device2newFunc[deviceKernel] = newFunc;
            deviceKernel->setSubprogram(nullptr);

            //copy old kernel over to the new
            ValueToValueMapTy VMap;
            auto NewFArgIt = newFunc->arg_begin();
            for (auto &Arg: deviceKernel->args()) {
              auto ArgName = Arg.getName();
              NewFArgIt->setName(ArgName);
              VMap[&Arg] = &(*NewFArgIt++);
            }

            errs() << "MergeKernel: created new Function: " << *(newFunc) << "\n";
            SmallVector<ReturnInst*, 8> Returns;
            llvm::CloneFunctionInto(newFunc, deviceKernel, VMap, false, Returns);
            errs() << *(newFunc) << "\n";

            ////remove the use of the argument in device kernel
            //std::vector<Instruction*> uses2remove;
            //for(auto arg = deviceKernel->arg_begin(); arg != deviceKernel->arg_end(); ++arg) {
            //  auto newArg = VMap[arg];
            //  for(User *U : newArg->users()){
            //    if(Instruction *UI = dyn_cast<Instruction>(U)){
            //      uses2remove.push_back(UI);
            //    }
            //  }
            //}
            //for(auto I : uses2remove)
            //  I->eraseFromParent();

            //set argument name in new kernel
            int i = 0;
            for(auto arg = newFunc->arg_begin(); arg != newFunc->arg_end(); ++arg) {
              bool foundOldArg = false;
              for(auto [oldArg, newArg] : VMap){
                if(newArg == arg){
                  foundOldArg = true;
                  break;
                }
              }
              if(foundOldArg) continue;

              switch(i){
                case 0:
                  arg->setName("gridDim.x");
                  break;
                case 1:
                  arg->setName("gridDim.y");
                  break;
                case 2:
                  arg->setName("gridDim.z");
                  break;
                case 3:
                  arg->setName("blockDim.x");
                  break;
                case 4:
                  arg->setName("blockDim.y");
                  break;
                case 5:
                  arg->setName("blockDim.z");
                  break;
                case 6:
                  arg->setName("blockIdx.x");
                  break;
                case 7:
                  arg->setName("blockIdx.y");
                  break;
                case 8:
                  arg->setName("blockIdx.z");
                  break;
                case 9:
                  arg->setName("threadIdx.x");
                  break;
                case 10:
                  arg->setName("threadIdx.y");
                  break;
                case 11:
                  arg->setName("threadIdx.z");
                  break;
              }
              ++i;
            }

            //replace the use of llvm.nvvm.read.ptx.sreg.* by new arguments
            //TODO: map doesn't accomodate multiple calls
            std::vector<Instruction*> cudaCall2remove;
            for(inst_iterator I = inst_begin(newFunc),
                E = inst_end(newFunc); I != E; ++I){
              CallInst *ci = dyn_cast<CallInst>(&*I);
              if(!ci) continue;
              auto calledFuncName = ci->getCalledFunction()->getName();

              std::map<std::string, std::string> nvvmCall2arg {
                { "llvm.nvvm.read.ptx.sreg.tid.x", "threadIdx.x"},
                { "llvm.nvvm.read.ptx.sreg.tid.y", "threadIdx.y"},
                { "llvm.nvvm.read.ptx.sreg.tid.z", "threadIdx.z"},
                { "llvm.nvvm.read.ptx.sreg.ntid.x", "blockDim.x"},
                { "llvm.nvvm.read.ptx.sreg.ntid.y", "blockDim.y"},
                { "llvm.nvvm.read.ptx.sreg.ntid.z", "blockDim.z"},
                { "llvm.nvvm.read.ptx.sreg.ctaid.x", "blockIdx.x"},
                { "llvm.nvvm.read.ptx.sreg.ctaid.y", "blockIdx.y"},
                { "llvm.nvvm.read.ptx.sreg.ctaid.z", "blockIdx.z"},
                { "llvm.nvvm.read.ptx.sreg.nctaid.x", "gridDim.x"},
                { "llvm.nvvm.read.ptx.sreg.nctaid.y", "gridDim.y"},
                { "llvm.nvvm.read.ptx.sreg.nctaid.z", "gridDim.z"}
              };

              if(nvvmCall2arg.find(calledFuncName) == nvvmCall2arg.end()) continue;
              auto argName = nvvmCall2arg[calledFuncName];

              cudaCall2remove.push_back(ci);
              for(auto arg = newFunc->arg_begin(); arg != newFunc->arg_end(); ++arg){
                if(arg->getName() != argName) continue;
                for(User *U : ci->users()){
                  Instruction *inst = dyn_cast<Instruction>(U);
                  if(!inst) continue;
                  for (auto OI = inst->op_begin(), OE = inst->op_end(); OI != OE; ++OI){
                    Value *val = *OI;
                    if(val == ci)
                      *OI = arg;
                  }
                }
              }
            }

            for(auto I : cudaCall2remove){
              errs() << "mergeKernel: cudaCall2remove: " << *I << "\n";
              errs() << "mergeKernel: what does I belong to?" <<  I->getParent()->getName();
              I->eraseFromParent();
            }
            insts2Remove.push_back(CI);
          }
          else if(calledFunc->getName().contains("cudaMalloc") && calledFunc->getName() != "cudaMalloc"){
            //auto devDataPtr = CI->getArgOperand(0);
            //auto AllocSize = CI->getArgOperand(1);
            //PointerType* Ty = dyn_cast<PointerType>(devDataPtr->getType());
            //PointerType* points2Ty = dyn_cast<PointerType>(Ty->getPointerElementType());
            //Instruction* Malloc = CallInst::CreateMalloc(CI,
            //                                   AllocSize->getType(), points2Ty->getPointerElementType(), AllocSize,
            //                                   nullptr, nullptr, "");
            //new StoreInst(Malloc, devDataPtr, CI);
            //errs() << "mergeKernel: replaced cuda malloc with: " << *Malloc << "\n";

            funcs2delete.insert(calledFunc);
            //for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
            //    for (Instruction::op_iterator I_Op = (&*I)->op_begin(), E_Op = (&*I)->op_end(); I_Op != E_Op; ++I_Op){
            //      if(*I_Op == devDataPtr){
            //        *I_Op = Malloc;
            //      }
            //    }
            //}
            insts2Remove.push_back(CI);
          }
          else if(calledFunc->getName().contains("cudaMemcpy")){
            ConstantInt* mode = dyn_cast<ConstantInt>(CI->getArgOperand(3));
            Instruction* originalBitcast = nullptr;
            mode->isOne()? originalBitcast = dyn_cast<BitCastInst>(CI->getArgOperand(1)) :
                           originalBitcast = dyn_cast<BitCastInst>(CI->getArgOperand(0));
            Instruction* devBitcast = nullptr;
            mode->isOne()? devBitcast = dyn_cast<BitCastInst>(CI->getArgOperand(0)) :
                           devBitcast = dyn_cast<BitCastInst>(CI->getArgOperand(1));
            assert(originalBitcast && devBitcast && "mergeKernel: didn't find bitcast from cudaMemcpy!\n");
            LoadInst *devLd = dyn_cast<LoadInst>(devBitcast->getOperand(0));
            LoadInst *originalLd = dyn_cast<LoadInst>(originalBitcast->getOperand(0));
            assert(originalLd && devLd && "mergeKernel: didn't find load from cudaMemcpy!\n");
            AllocaInst *devAlloc = dyn_cast<AllocaInst>(devLd->getOperand(0));
            AllocaInst *originalAlloc = dyn_cast<AllocaInst>(originalLd->getOperand(0));
            assert(devAlloc && originalAlloc && "mergeKernel: didn't find alloca from cudaMemcpy!\n");
            std::map<AllocaInst*, LoadInst>originalAlloc2newLD;
            auto newLd = new LoadInst(cast<PointerType>(originalAlloc->getType())->getElementType(), originalAlloc, "ldHost", originalLd);
            Value* cpySize = nullptr;
            errs() << "mergeKernel: found originalAlloc " << *originalAlloc << "\n";
            for(auto user : originalAlloc->users()){
              if(StoreInst *st = dyn_cast<StoreInst>(user)){
              }
            }
            for(auto user : originalAlloc->users()){
              errs() << "mergeKernel: found originalAlloc user" << *user << "\n";
              if(StoreInst *st = dyn_cast<StoreInst>(user)){
                errs() << "mergeKernel: found originalAlloc store:" << *st << "\n";
                if(BitCastInst* cast = dyn_cast<BitCastInst>(st->getOperand(0))){
                  errs() << "mergeKernel: found originalAlloc cast:" << *cast << "\n";
                  if(CallInst* ci = dyn_cast<CallInst>(cast->getOperand(0))){
                    errs() << "mergeKernel: found originalAlloc ci:" << *ci << "\n";
                    if(!ci->getCalledFunction()->getName().contains("malloc")) continue;
                    cpySize = ci->getArgOperand(0);
                    LLVMContext &C = cpySize->getContext();
                    MDNode *mdSize = nullptr;
                    if(Instruction* sizeInst = dyn_cast<Instruction>(cpySize)){
                      std::string mdDataSize = "tulip.target.datasize";
                      mdSize = MDNode::get(C, MDString::get(C, std::to_string(mdID)));
                      sizeInst->setMetadata("tulip.target.datasize", mdSize);
                      mdID++;
                    }
                    MDNode* N;
                    mdSize ? N = MDNode::get(C, mdSize) :
                             N = MDNode::get(C, ValueAsMetadata::get(dyn_cast<ConstantInt>(cpySize)));
                    mode->isOne() ? ci->setMetadata("tulip.target.mapdata.to", N) :
                                    ci->setMetadata("tulip.target.mapdata.from", N);
                  }
                }
              }
            }
            for(auto user : devAlloc->users()){
              if(LoadInst *ld = dyn_cast<LoadInst>(user)){
                for(auto user : ld->users()){
                  Instruction* UI = dyn_cast<Instruction>(user);
                  for (auto OI = UI->op_begin(), OE = UI->op_end(); OI != OE; ++OI){
                    Value *val = *OI;
                    if(val == ld)
                      *OI = newLd;
                  }
                }
              }
            }




            //TODO: better to create an empty function that doesn't really do anything
            if(mode->isOne()){

              bool foundMapRegion = false;
              for(auto &I : *(CI->getParent())){
                if(I.getMetadata("tulip.target.start.of.map")){
                  foundMapRegion = true;
                  insts2Remove.push_back(CI);
                  break;
                }
              }
              if(foundMapRegion) continue;
              LLVMContext &C = CI->getContext();
              MDNode *N = MDNode::get(C, MDString::get(C,""));
              CI->setMetadata("tulip.target.start.of.map", N);
            }
            else{
              bool foundMapRegion = false;
              for(auto &I : *(CI->getParent())){
                if(I.getMetadata("tulip.target.end.of.map")){
                  foundMapRegion = true;
                  insts2Remove.push_back(CI);
                  break;
                }
              }
              if(foundMapRegion) continue;
              LLVMContext &C = CI->getContext();
              MDNode *N = MDNode::get(C, MDString::get(C,""));
              CI->setMetadata("tulip.target.end.of.map", N);
            }
            //errs() << "mergeKernel: cudaMemcpy: " << *CI << "\n";
            //Type* Int1Ty = Type::getInt1Ty(F->getContext());
            //CallSite CS(CI);
            //SmallVector<Value *, 4> Args(CS.arg_begin(), CS.arg_end()-1);
            //Args.push_back(ConstantInt::getFalse(Int1Ty));
            //ArrayRef<Value*> args(Args);
            //std::vector<Type*> argTyVec;
            //argTyVec.push_back(PointerType::get(Type::getInt8Ty(F->getContext()), 0));
            //argTyVec.push_back(PointerType::get(Type::getInt8Ty(F->getContext()), 0));
            //argTyVec.push_back(Type::getInt64Ty(F->getContext()));
            //argTyVec.push_back(Int1Ty);
            //ArrayRef<Type *> argTys(argTyVec);
            //FunctionType* memcpyFuncTy = FunctionType::get(
            //    Type::getVoidTy(F->getContext()), //return type
            //    argTys,
            //    false
            //);

            //auto MemCpyFunc = F->getParent()->getOrInsertFunction("llvm.memcpy.p0i8.p0i8.i64", memcpyFuncTy);
            //CallInst *NewCI = CallInst::Create(MemCpyFunc, args, "", CI);

            ////add metadata to identify omp target mapping
            ////Instruction* src = CI->getArgOperand(1);
            //Value* cpySize = CI->getArgOperand(2);
            //ConstantInt* mode = dyn_cast<ConstantInt>(CI->getArgOperand(3));
            //Instruction* dest = nullptr;
            //mode->isOne()? dest = dyn_cast<Instruction>(CI->getArgOperand(0)) :
            //               dest = dyn_cast<Instruction>(CI->getArgOperand(1));
            //LLVMContext& C = NewCI->getContext();
            //std::string mdDevice = "tulip.target.mapdata";
            //MDNode *mdSize = nullptr;
            //if(Instruction* sizeInst = dyn_cast<Instruction>(cpySize)){
            //  std::string mdDataSize = "tulip.target.datasize";
            //  mdSize = MDNode::get(C, MDString::get(C, std::to_string(mdID)));
            //  sizeInst->setMetadata("tulip.target.datasize", mdSize);
            //  mdID++;
            //}

            //MDNode* N;
            //mdSize ? N = MDNode::get(C, mdSize) :
            //         N = MDNode::get(C, ValueAsMetadata::get(dyn_cast<ConstantInt>(cpySize)));
            //mode->isOne() ? dest->setMetadata("tulip.target.mapdata.to", N) :
            //                dest->setMetadata("tulip.target.mapdata.from", N);
          }
          else if(calledFunc->getName().contains("cudaDeviceSynchronize")){
            insts2Remove.push_back(CI);
          }
        }
      }


      //create loops around kernel
      for(auto [configureCI, kernelProfile] : kernelProfiles){
        std::stack<BasicBlock*> headerNests, headerNests2;
        auto kernelBB = kernelProfile->kernelBB;
        BasicBlock* pred = kernelBB->getSinglePredecessor();
        std::vector<BasicBlock*> succs(succ_begin(kernelBB), succ_end(kernelBB));
        auto term = dyn_cast<BranchInst>(pred->getTerminator());
        assert(term && "mergeKernel: term is not a branch inst 206\n");
        auto kernelPred = kernelBB->getSinglePredecessor();
        assert(kernelPred && "mergeKernel: kernel has multiple predecessors\n");
        insts2Remove.push_back(term);
        Value *cond = nullptr;
        if(term->isConditional())
          cond = term->getCondition();
        int loopCnt = 0;
        BasicBlock *lastheader = nullptr;

        //create headers
        std::map<BasicBlock*, Value*>header2itNum;
        auto loopDims = kernelProfile->loopDims;
        for(auto itNum : loopDims){
          if(ConstantInt *constInt = dyn_cast<ConstantInt>(itNum))
            if(constInt->getSExtValue() == 1)
              continue;
          auto header = BasicBlock::Create(kernelBB->getContext(), "header." + std::to_string(loopCnt), F, kernelBB);
          headerNests.push(header);
          header2itNum[header] = itNum;
          if(loopCnt == 0){
            auto br = BranchInst::Create(header, pred);
          //  LLVMContext& C = term->getContext();
          //  MDNode* N = MDNode::get(C, MDString::get(C, ""));
          //  term->setMetadata("splendid.doall.loop", N);
          }
          pred = header;
          loopCnt++;
          lastheader = header;
        }
        headerNests2 = headerNests;

        //create latches
        BasicBlock* lastLatch = nullptr;
        std::map<BasicBlock*, BasicBlock*> header2latch;
        for(int i=loopCnt-1; i>=0; --i){
          auto header = headerNests.top();
          auto latch = BasicBlock::Create(kernelBB->getContext(), "latch." + std::to_string(i), F, kernelBB);
          if(!lastLatch)
            lastLatch = latch;
          auto br = BranchInst::Create(header, latch);

          header2latch[header] = latch;
          headerNests.pop();
        }


        //create header branches
        auto loopExit = kernelBB->getSingleSuccessor();
        assert(loopExit && "kernel BB has multiple exits\n");

        BasicBlock *prevHeader = nullptr;
        int i = loopCnt-1;
        std::vector<Value*> indvars;
        while(!headerNests2.empty()){
          auto header = headerNests2.top();
          headerNests2.pop();
          auto nextHeader = headerNests2.empty()? nullptr : headerNests2.top();

          //create phi node
          Type *phiTy = header2itNum[header]->getType();
          auto indvar = PHINode::Create(phiTy, 2, "indvar."+std::to_string(i), header);
          indvars.push_back(indvar);
          CmpInst *cmp = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_ULT, indvar, header2itNum[header], "exitCheck." + std::to_string(i), header);
          BranchInst *term = nullptr;

          //create increment in latch
          auto latch = header2latch[header];
          Value *incr = BinaryOperator::Create(Instruction::Add, indvar, ConstantInt::get(phiTy, 1),
                                    "indvar.next." + std::to_string(i), latch->getTerminator());

          //TODO: figure out why i need getName empty

          //a single loop
          if(!nextHeader && !prevHeader){
            term = BranchInst::Create(kernelBB, loopExit, cmp, header);
            indvar->addIncoming(ConstantInt::get(phiTy,0), kernelPred);
          }
          else if(!nextHeader || !nextHeader->hasName() || nextHeader->getName() == ""){
            term = BranchInst::Create(prevHeader, loopExit, cmp, header);
            indvar->addIncoming(ConstantInt::get(phiTy,0), kernelPred);
          }
          else if(!prevHeader){
            term = BranchInst::Create(kernelBB, header2latch[nextHeader], cmp, header);
            indvar->addIncoming(ConstantInt::get(phiTy,0), nextHeader);
          }
          else{
            term = BranchInst::Create(prevHeader, header2latch[nextHeader], cmp, header);
            indvar->addIncoming(ConstantInt::get(phiTy,0), nextHeader);
          }

          if(kernelProfile->dim2classify[header2itNum[header]] == 1 ||
              kernelProfile->dim2classify[header2itNum[header]] == 2){
            LLVMContext& C = term->getContext();
            MDNode* N = MDNode::get(C, MDString::get(C, ""));
            if(kernelProfile->dim2classify[header2itNum[header]] == 1){
              if(kernelProfile->gridLoopCnt > 1) term->setMetadata("tulip.doall.loop.grid.collapse", N);
              else term->setMetadata("tulip.doall.loop.grid", N);
            }
            if(kernelProfile->dim2classify[header2itNum[header]] == 2){
              if(kernelProfile->blockLoopCnt > 1) term->setMetadata("tulip.doall.loop.block.collapse", N);
              else term->setMetadata("tulip.doall.loop.block", N);
            }
            errs() << "mergeKernel: create metadata" << *term << "\n";
          }
          indvar->addIncoming(incr, header2latch[header]);
          prevHeader = header;
          i--;
        }

        //replace kernel branch
        insts2Remove.push_back(kernelBB->getTerminator());
        BranchInst::Create(lastLatch, kernelBB);



        std::vector<Value*> newKernelArgs;
        auto kernelCall = kernelProfile->kernelCall;
        for(int i=0; i<kernelCall->arg_size(); ++i){
          errs() << "SUSAN: original arg " << *(kernelCall->getArgOperand(i)) << "\n";
          newKernelArgs.push_back(kernelCall->getArgOperand(i));
        }
        std::reverse(indvars.begin(), indvars.end());
        std::vector<Value*> indvarsExtended; i=0;
        auto i32Ty = Type::getInt32Ty(kernelCall->getContext());
        if(loopDims.size() == 2){
          //push dims
          auto dim = loopDims[0];
          auto arg = dim;
          if(!dim->getType()->isIntegerTy(32))
            arg = CastInst::CreateTruncOrBitCast(dim, i32Ty, "dim.cast", kernelCall);
          newKernelArgs.push_back(arg);
          newKernelArgs.push_back(ConstantInt::get(i32Ty, 1));
          newKernelArgs.push_back(ConstantInt::get(i32Ty, 1));
          dim = loopDims[1];
          arg = dim;
          if(!dim->getType()->isIntegerTy(32))
            arg = CastInst::CreateTruncOrBitCast(dim, i32Ty, "dim.cast", kernelCall);
          newKernelArgs.push_back(arg);
          newKernelArgs.push_back(ConstantInt::get(i32Ty, 1));
          newKernelArgs.push_back(ConstantInt::get(i32Ty, 1));


          //create indvars
          auto indvar = indvars[0];
          arg = indvar;
          if(!arg->getType()->isIntegerTy(32))
            arg = CastInst::CreateTruncOrBitCast(indvar, i32Ty, "dim.cast", kernelCall);
          indvarsExtended.push_back(arg);
          indvarsExtended.push_back(ConstantInt::get(i32Ty, 0));
          indvarsExtended.push_back(ConstantInt::get(i32Ty, 0));
          indvar = indvars[1];
          arg = indvar;
          if(!arg->getType()->isIntegerTy(32))
            arg = CastInst::CreateTruncOrBitCast(indvar, i32Ty, "dim.cast", kernelCall);
          indvarsExtended.push_back(arg);
          indvarsExtended.push_back(ConstantInt::get(Type::getInt32Ty(kernelCall->getContext()), 0));
          indvarsExtended.push_back(ConstantInt::get(Type::getInt32Ty(kernelCall->getContext()), 0));
        } else {
          for(auto itNum : loopDims){
            errs() << "SUSAN: itnum: " << *itNum << "\n";
            newKernelArgs.push_back(itNum);
            if(ConstantInt *constInt = dyn_cast<ConstantInt>(itNum))
              if(constInt->getSExtValue() == 1){
                indvarsExtended.push_back(ConstantInt::get(Type::getInt32Ty(kernelCall->getContext()), 0));
                continue;
              }
            indvarsExtended.push_back(indvars[i]);
            ++i;
          }
        }

        for(auto indvar : indvarsExtended){
          errs() << "mergeKernel: indvar " << *indvar << "\n";
          newKernelArgs.push_back(indvar);
        }
        auto newFunc = kernelProfile->newFunc;
        errs() << *(newFunc->getFunctionType()) << "\n";

        for(auto arg : newKernelArgs){
          errs() << "mergeKernel: new function kernel Args: " << *arg << "\n";
        }

        CallInst *newKernelCall = CallInst::Create(
              newFunc->getFunctionType(), //function type
              newFunc, //function
              ArrayRef<Value*>(newKernelArgs), //args
              "",
              kernelCall //insert before
            );
          insts2Remove.push_back(kernelCall);
      }

       //delete cuda calls and control flows
       for(auto I : insts2Remove)
         I->eraseFromParent();

    }

    //delete functions
    for(auto f : funcs2delete)
      f->eraseFromParent();

    //process shared variables
     std::map<Function*, std::set<Instruction*>> func2SharedMems;
     std::map<Function*, std::set<GlobalVariable*>> func2SharedGlobs;
     std::map<GlobalVariable*, std::set<Instruction*>> glob2Insts;
     std::map<Instruction*, User::op_iterator> inst2opIt;
     std::map<Instruction*, GEPOperator*> inst2gepOp;
     for (Module::global_iterator I = M.global_begin(), E = M.global_end();
           I != E; ++I) {
         GlobalVariable* globVal = &*I;
         if(!globVal->hasInitializer()) continue;
         if(globVal->getAddressSpace() != 3) continue;
         for(User *U : globVal->users()){
           ConstantExpr *UE = dyn_cast<ConstantExpr>(U);
           errs() << "UE: " << *UE << "\n";
           if(!UE) continue;

           //find instruction that uses the expr
           for(User *ExprU : UE->users()){
              Instruction* UI = dyn_cast<Instruction>(ExprU);
              errs() << "ExprU: " << *ExprU << "\n";
              if(!UI){
                auto gepExpr = dyn_cast<GEPOperator>(ExprU);
                if(!gepExpr) continue;
                for(auto gepU : gepExpr->users()){
                  auto LI = dyn_cast<Instruction>(gepU);
                  if(!LI) continue;
                  inst2gepOp[LI] = gepExpr;
                }
                continue;
              }
              Function *func = UI->getParent()->getParent();
              func2SharedMems[func].insert(UI);
              glob2Insts[globVal].insert(UI);
              func2SharedGlobs[func].insert(globVal);

              for (auto OI = UI->op_begin(), OE = UI->op_end(); OI != OE; ++OI){
                Value *val = *OI;
                if(val == UE)
                  inst2opIt[UI] = OI;
              }
           }
         }
     }
     for(auto [func, sharedGlobs] : func2SharedGlobs){
       const DataLayout &DL = M.getDataLayout();
       int i=0;
       for(auto glob : sharedGlobs){
          PointerType* ty = dyn_cast<PointerType>(glob->getType());
          if(!ty){
            errs() << "WARNINGS: shared object ty is not a pointer type\n";
            continue;
          }

          Type* pointedTy = ty->getPointerElementType();
          auto &entryBB = func->getEntryBlock();
          auto term = entryBB.getTerminator();
          AllocaInst *sharedAlloc = new AllocaInst(pointedTy, 0, nullptr, "sharedMem"+std::to_string(i), &(entryBB.front()));
          ++i;
          for(auto globInst : glob2Insts[glob]){
            if(globInst->getParent()->getParent() != func) continue;
            if(inst2opIt.find(globInst) != inst2opIt.end())
              *(inst2opIt[globInst]) = sharedAlloc;
            errs() << "gepinst: " << *globInst << "\n";
          }
          for(auto [LI, gepOp] : inst2gepOp){
            if(LI->getParent()->getParent() != func) continue;
            std::vector<Value*> idxList(gepOp->idx_begin(), gepOp->idx_end());
            auto gepInst = GetElementPtrInst::Create(
                  gepOp->getSourceElementType(),
                  sharedAlloc,
                  makeArrayRef(idxList),
                  "sharedMem.gep" + std::to_string(i),
                  LI
                );
            for (auto OI = LI->op_begin(), OE = LI->op_end(); OI != OE; ++OI){
              if(*OI == gepOp)
                *OI = gepInst;
            }

          }

       }
     }
     for(auto [func, sharedMemVars] : func2SharedMems){
       errs() << "mergeKernel: func that contain shared mem objects: " << func->getName() << "\n";
       for(auto var : sharedMemVars)
         errs() << "mergeKernel: sharedMemvar: " << *var << "\n";
     }

    return false;
  }
}; // end of struct Hello
}  // end of anonymous namespace

char MergeKernel::ID = 0;
static RegisterPass<MergeKernel> X("merge-kernel", "merge cuda kernel back to main file",
                             false /* Only looks at CFG */,
                             false /* Analysis Pass */);
