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
struct MergeKernel : public ModulePass {
  static char ID;
  std::vector<Value*> loopDims;
  std::vector<Value*> indvars;
  MergeKernel() : ModulePass(ID) {}

  void findThreadDim(Function &F, LoadInst *DimArg, bool blockOrGrid){
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
        loopDims.push_back(dim);
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
          loopDims.push_back(dim);
        }
        foundDim = true;
        break;
      }
    }
  }

  bool runOnModule(Module &M) override {
  std::vector<Function*>funcs2delete;
  for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
    Function *F = &*FI;
    F->removeFnAttr("target-features");
    F->removeFnAttr("target-cpu");
    loopDims.clear();
    BasicBlock *kernelBB = nullptr;
    std::vector<Instruction*> insts2Remove;
    CallInst *kernelCall = nullptr;
    Function *newFunc = nullptr;
    Function *deviceKernel = nullptr;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if(CallInst *CI = dyn_cast<CallInst>(&*I)){
        Function* calledFunc = CI->getCalledFunction();
        if(calledFunc->getName().contains("_ZN4dim3C2Ejjj")){
          insts2Remove.push_back(CI);
        }
        else if(calledFunc->getName().contains("cudaConfigureCall")){
          errs() << "CudaFE: found cudaConfigureCall\n";

          //remove control flow caused by configuration failure
          BranchInst *CF2Remove = dyn_cast<BranchInst>(CI->getParent()->getTerminator());
          CmpInst *cmp = dyn_cast<CmpInst>(CF2Remove->getCondition());
          if(cmp){
            if(cmp->getPredicate() == CmpInst::ICMP_NE){
              auto opnd0 = cmp->getOperand(0);
              auto opnd1 = cmp->getOperand(1);
              bool ConfigureThenBranchPattern = false;
              if(ConstantInt *integer = dyn_cast<ConstantInt>(opnd0)){
                if(integer->getZExtValue() == 0 && opnd1 == CI)
                  ConfigureThenBranchPattern = true;
              } else if(ConstantInt *integer = dyn_cast<ConstantInt>(opnd1)){
                if(integer->getZExtValue() == 0 && opnd0 == CI)
                  ConfigureThenBranchPattern = true;
              }
              if(ConfigureThenBranchPattern){
                CF2Remove->setCondition(ConstantInt::get(Type::getInt1Ty(CF2Remove->getContext()), 0));
                kernelBB = CF2Remove->getSuccessor(1);

                //write metadata to the function call
                for(auto &I : *kernelBB){
                  if(CallInst *CI = dyn_cast<CallInst>(&I)){
                    Function *calledFunc = CI->getCalledFunction();
                    for (inst_iterator I = inst_begin(calledFunc), E = inst_end(calledFunc); I != E; ++I) {
                      if(CallInst *ci = dyn_cast<CallInst>(&*I)){
                        Function *calledF = ci->getCalledFunction();
                        if(calledF->getName().contains("cudaLaunch")){
                          kernelCall = CI;
                          break;
                        }
                      }
                    }
                  }

                  if(kernelCall)
                    break;
                }

                assert(kernelCall && "mergeKernel: din't find kernel call!\n");
                errs() << "mergeKernel: kernel call: " << *kernelCall << "\n";


                insts2Remove.push_back(cmp);
              }
            }
          }

          errs() << "MergeKernel: blocks per grid:\n";
          findThreadDim(*F, dyn_cast<LoadInst>(CI->getArgOperand(0)), false);
          errs() << "MergeKernel: threads per block:\n";
          findThreadDim(*F, dyn_cast<LoadInst>(CI->getArgOperand(2)), true);

          //find host kernel function and actual kernel name
          Module *M = F->getParent();
          auto hostKernel = kernelCall->getCalledFunction();
          funcs2delete.push_back(hostKernel);
          StringRef host_kernelName = hostKernel->getName();
          auto namePair = host_kernelName.rsplit("_CudaFE_");
          StringRef kernelName = namePair.second;
          errs() << "mergeKernel: found kernelName: " << kernelName << "\n";

          //Find device kernel function
          for (Module::iterator FI = M->begin(), FE = M->end(); FI != FE; ++FI) {
            Function *F = &*FI;
            auto funcName = F->getName();
            if(funcName.contains(kernelName) &&
               funcName != host_kernelName ){
              deviceKernel = F;
              funcs2delete.push_back(deviceKernel);
              break;
            }
          }
          errs() << "MergeKernel: found deviceKernel: " << deviceKernel->getName() << "\n";



          //create a new function for kernel
          Type* i32Ty = Type::getInt32Ty(deviceKernel->getContext());
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
          newFunc = Function::Create(
                funcTy,
                deviceKernel->getLinkage(),
                deviceKernel->getName(),
                *M
              );
          errs() << "MergeKernel: created new Function: " << *newFunc << "\n";

          //copy old kernel over to the new
          ValueToValueMapTy VMap;
          auto NewFArgIt = newFunc->arg_begin();
          for (auto &Arg: deviceKernel->args()) {
            auto ArgName = Arg.getName();
            NewFArgIt->setName(ArgName);
            VMap[&Arg] = &(*NewFArgIt++);
          }
          SmallVector<ReturnInst*, 8> Returns;
          llvm::CloneFunctionInto(newFunc, deviceKernel, VMap, false, Returns);
          errs() << *newFunc << "\n";

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
            CallInst *CI = dyn_cast<CallInst>(&*I);
            if(!CI) continue;
            auto calledFuncName = CI->getCalledFunction()->getName();
            if(calledFuncName == "llvm.nvvm.read.ptx.sreg.tid.x")
              arg2CI["threadIdx.x"] = CI;
            else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.tid.y")
              arg2CI["threadIdx.y"] = CI;
            else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.tid.z")
              arg2CI["threadIdx.z"] = CI;
            else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.ntid.x")
              arg2CI["blockDim.x"] = CI;
            else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.ntid.y")
              arg2CI["blockDim.y"] = CI;
            else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.ntid.z")
              arg2CI["blockDim.z"] = CI;
            else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.ctaid.x")
              arg2CI["blockIdx.x"] = CI;
            else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.ctaid.y")
              arg2CI["blockIdx.y"] = CI;
            else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.ctaid.z")
              arg2CI["blockIdx.z"] = CI;
            else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.nctaid.x")
              arg2CI["gridDim.x"] = CI;
            else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.nctaid.y")
              arg2CI["gridDim.y"] = CI;
            else if(calledFuncName == "llvm.nvvm.read.ptx.sreg.nctaiid.z")
              arg2CI["gridDim.z"] = CI;
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

          errs() << *newFunc << "\n";
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

          errs() << *newFunc << "\n";



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
    if(kernelBB){
      std::stack<BasicBlock*> headerNests, headerNests2;
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
      for(auto itNum : loopDims){
        if(ConstantInt *constInt = dyn_cast<ConstantInt>(itNum))
          if(constInt->getSExtValue() == 1)
            continue;
        auto header = BasicBlock::Create(kernelBB->getContext(), "header." + std::to_string(loopCnt), F, kernelBB);
        headerNests.push(header);
        header2itNum[header] = itNum;
        if(loopCnt == 0)
          BranchInst::Create(header, pred);
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
        BranchInst::Create(header, latch);
        header2latch[header] = latch;
        headerNests.pop();
      }


      //create header branches
      auto loopExit = kernelBB->getSingleSuccessor();
      assert(loopExit && "kernel BB has multiple exits\n");

      BasicBlock *prevHeader = nullptr;
      int i = loopCnt-1;
      while(!headerNests2.empty()){
        auto header = headerNests2.top();
        headerNests2.pop();
        auto nextHeader = headerNests2.top();

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
        if(!nextHeader || nextHeader->getName() == ""){
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
        indvar->addIncoming(incr, header2latch[header]);
        prevHeader = header;
        i--;
      }

      //replace kernel branch
      insts2Remove.push_back(kernelBB->getTerminator());
      BranchInst::Create(lastLatch, kernelBB);

    }


    //replace kernel call
    if(kernelCall){
    std::vector<Value*> newKernelArgs;
    for(int i=0; i<kernelCall->arg_size(); ++i){
      errs() << "SUSAN: original arg " << *kernelCall->getArgOperand(i) << "\n";
      newKernelArgs.push_back(kernelCall->getArgOperand(i));
    }
    std::reverse(indvars.begin(), indvars.end());
    std::vector<Value*> indvarsExtended; int i=0;
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
    for(auto indvar : indvarsExtended){
      errs() << "SUSAN: indvar " << *indvar << "\n";
      newKernelArgs.push_back(indvar);
    }
    errs() << *(newFunc->getFunctionType()) << "\n";
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
