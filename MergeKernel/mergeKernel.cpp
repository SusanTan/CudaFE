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

using namespace llvm;

namespace {
class KernelProfile{
  public:
  std::vector<Value*> loopDims;
  BasicBlock *kernelBB = nullptr;
  Function *deviceKernel = nullptr;
  CallInst *kernelCall = nullptr;
  Function *newFunc = nullptr;
};

struct MergeKernel : public ModulePass {
  static char ID;
  std::vector<Value*> loopDims;
  std::set<Function*>funcs2delete;
  MergeKernel() : ModulePass(ID) {}

  void findThreadDim(KernelProfile *kernelProfile, Function &F, LoadInst *DimArg, bool blockOrGrid){
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
        }
        foundDim = true;
        break;
      }
    }
  }

  bool runOnModule(Module &M) override {
    for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
      std::map<CallInst*, KernelProfile*> kernelProfiles;
      Function *F = &*FI;
      F->removeFnAttr("target-features");
      F->removeFnAttr("target-cpu");
      loopDims.clear();
      BasicBlock *kernelBB = nullptr;
      std::vector<Instruction*> insts2Remove;
      for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
        if(CallInst *CI = dyn_cast<CallInst>(&*I)){
          Function* calledFunc = CI->getCalledFunction();
          if(calledFunc->getName().contains("_ZN4dim3C2Ejjj")){
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
            auto ptr = CI->getArgOperand(0);
            CallInst::CreateFree(ptr, CI);
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
            }
            else{
              findThreadDim(kernelProfiles[CI], *F, dyn_cast<LoadInst>(CI->getArgOperand(0)), false);
            }

            errs() << "MergeKernel: threads per block:\n";
            if(isa<ConstantInt>(CI->getArgOperand(2))){
              Value *dim = CI->getArgOperand(2);
              kernelProfiles[CI]->loopDims.push_back(dim);
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

            //replace the use of llvm.nvvm.read.ptx.sreg.* by new arguments
            std::map<std::string, Instruction*> arg2CI;
            for(inst_iterator I = inst_begin(newFunc),
                E = inst_end(newFunc); I != E; ++I){
              CallInst *ci = dyn_cast<CallInst>(&*I);
              if(!ci) continue;
              auto calledFuncName = ci->getCalledFunction()->getName();
              if(calledFuncName == "llvm.nvvm.read.ptx.sreg.tid.x")
                arg2CI["threadIdx.x"] = ci;
              else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.tid.y")
                arg2CI["threadIdx.y"] = ci;
              else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.tid.z")
                arg2CI["threadIdx.z"] = ci;
              else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.ntid.x")
                arg2CI["blockDim.x"] = ci;
              else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.ntid.y")
                arg2CI["blockDim.y"] = ci;
              else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.ntid.z")
                arg2CI["blockDim.z"] = ci;
              else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.ctaid.x")
                arg2CI["blockIdx.x"] = ci;
              else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.ctaid.y")
                arg2CI["blockIdx.y"] = ci;
              else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.ctaid.z")
                arg2CI["blockIdx.z"] = ci;
              else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.nctaid.x")
                arg2CI["gridDim.x"] = ci;
              else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.nctaid.y")
                arg2CI["gridDim.y"] = ci;
              else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.nctaiid.z")
                arg2CI["gridDim.z"] = ci;
            }

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

            //remove cuda function call
            std::vector<Instruction*> cudaCall2remove;
            for(auto arg = newFunc->arg_begin(); arg != newFunc->arg_end(); ++arg) {
              if(arg2CI.find(arg->getName()) == arg2CI.end()) continue;
              cudaCall2remove.push_back(arg2CI[arg->getName()]);
              for(User *U : arg2CI[arg->getName()]->users()){
                Instruction *inst = dyn_cast<Instruction>(U);
                if(!inst) continue;
                for (auto OI = inst->op_begin(), OE = inst->op_end(); OI != OE; ++OI){
                  Value *val = *OI;
                  if(val == arg2CI[arg->getName()])
                    *OI = arg;
                }
              }
            }
            for(auto I : cudaCall2remove)
              I->eraseFromParent();
            insts2Remove.push_back(CI);
          }
          else if(calledFunc->getName().contains("_ZL10cudaMallocIdE9cudaErrorPPT_m")){
            auto devDataPtr = CI->getArgOperand(0);
            auto AllocSize = CI->getArgOperand(1);
            PointerType* Ty = dyn_cast<PointerType>(devDataPtr->getType());
            PointerType* points2Ty = dyn_cast<PointerType>(Ty->getPointerElementType());
            Instruction* Malloc = CallInst::CreateMalloc(CI,
                                               AllocSize->getType(), points2Ty->getPointerElementType(), AllocSize,
                                               nullptr, nullptr, "");
            new StoreInst(Malloc, devDataPtr, CI);
            errs() << "mergeKernel: replaced cuda malloc with: " << *Malloc << "\n";

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
            Type* Int1Ty = Type::getInt1Ty(F->getContext());
            CallSite CS(CI);
            SmallVector<Value *, 4> Args(CS.arg_begin(), CS.arg_end()-1);
            Args.push_back(ConstantInt::getFalse(Int1Ty));
            ArrayRef<Value*> args(Args);
            std::vector<Type*> argTyVec;
            argTyVec.push_back(PointerType::get(Type::getInt8Ty(F->getContext()), 0));
            argTyVec.push_back(PointerType::get(Type::getInt8Ty(F->getContext()), 0));
            argTyVec.push_back(Type::getInt64Ty(F->getContext()));
            argTyVec.push_back(Int1Ty);
            ArrayRef<Type *> argTys(argTyVec);
            FunctionType* memcpyFuncTy = FunctionType::get(
                Type::getVoidTy(F->getContext()), //return type
                argTys,
                false
            );

            auto MemCpyFunc = F->getParent()->getOrInsertFunction("llvm.memcpy.p0i8.p0i8.i64", memcpyFuncTy);
            CallInst *NewCI = CallInst::Create(MemCpyFunc, args, "", CI);
            insts2Remove.push_back(CI);
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
          if(loopCnt == 0)
            auto br = BranchInst::Create(header, pred);
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
          errs() << "mergeKernel: nextHeader: "<< nextHeader << "\n";
          if(!nextHeader || !nextHeader->hasName() || nextHeader->getName() == ""){
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

          if(i == 0){
            LLVMContext& C = term->getContext();
            MDNode* N = MDNode::get(C, MDString::get(C, ""));
            term->setMetadata("splendid.doall.loop", N);
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

    return false;
  }
}; // end of struct Hello
}  // end of anonymous namespace

char MergeKernel::ID = 0;
static RegisterPass<MergeKernel> X("merge-kernel", "merge cuda kernel back to main file",
                             false /* Only looks at CFG */,
                             false /* Analysis Pass */);
