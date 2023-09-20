#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Module.h"


#include <set>

using namespace llvm;

namespace {
struct MergeKernel : public FunctionPass {
  static char ID;
  MergeKernel() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    std::vector<Instruction*> insts2Remove;
    for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
      if(CallInst *CI = dyn_cast<CallInst>(&*I)){
        Function* calledFunc = CI->getCalledFunction();
        if(calledFunc->getName() == "cudaConfigureCall"){
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
              errs() << "mergeKernel: configure then branch pattern found? " << ConfigureThenBranchPattern << "\n";
              if(ConfigureThenBranchPattern){
                CF2Remove->setCondition(ConstantInt::get(Type::getInt1Ty(CF2Remove->getContext()), 0));
                insts2Remove.push_back(cmp);
              }
            }
          }

          //find gridDim
          LoadInst *gridDimArg = dyn_cast<LoadInst>(CI->getArgOperand(0));
          assert(gridDimArg && "mergeKernel: gridDimArg is not a load inst\n");
          GetElementPtrInst *gridDimGep = dyn_cast<GetElementPtrInst>(gridDimArg->getOperand(0));
          assert(gridDimGep && "mergeKernel: gridDimGep is not a gep inst\n");
          AllocaInst *gridDimAllocaCoerce = dyn_cast<AllocaInst>(gridDimGep->getOperand(0));
          assert(gridDimAllocaCoerce && "mergeKernel: gridDimAllocaCoerce is not an alloca inst\n");
          errs() << "SUSAN: gridDimAllocaCoerce: " << *gridDimAllocaCoerce << "\n";
          BitCastInst *gridDimBitCast = nullptr;
          for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
            BitCastInst *bitcast = dyn_cast<BitCastInst>(&*I);
            if(!bitcast) continue;
            if(bitcast->getOperand(0) != gridDimAllocaCoerce) continue;
            gridDimBitCast = bitcast;
            break;
          }
          assert(gridDimBitCast && "mergeKernel: gridDimBitCast is not a bit cast inst\n");
          errs() << "SUSAN: gridDimBitCast: " << *gridDimBitCast << "\n";
          BitCastInst *gridDimBitCastStore = nullptr;
          for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
            CallInst *CI = dyn_cast<CallInst>(&*I);
            if(!CI) continue;
            if(CI->getCalledFunction()->getName() != "llvm.memcpy.p0i8.p0i8.i64") continue;
            if(CI->getArgOperand(0) != gridDimBitCast) continue;
            BitCastInst *bitcast = dyn_cast<BitCastInst>(CI->getArgOperand(1));
            gridDimBitCastStore = bitcast;
            break;
          }
          assert(gridDimBitCastStore && "mergeKernel: gridDimBitCastStore is not a bit cast store inst \n");
          AllocaInst *gridDimAlloca = dyn_cast<AllocaInst>(gridDimBitCastStore->getOperand(0));
          assert(gridDimAlloca && "mergeKernel: gridDimAlloca is not an alloca inst \n");
          Value *gridDim = nullptr;
          for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
            CallInst *CI = dyn_cast<CallInst>(&*I);
            if(!CI) continue;
            if(CI->getCalledFunction()->getName() != "_ZN4dim3C2Ejjj") continue;
            if(CI->getArgOperand(0) != gridDimAlloca) continue;
            Value *dim1 = CI->getArgOperand(1);
            Value *dim2 = CI->getArgOperand(2);
            Value *dim3 = CI->getArgOperand(3);
            errs() << "mergeKernel: grid Dim 1: " << *dim1 <<"\n";
            break;
          }




          insts2Remove.push_back(CI);
        }
        else if(calledFunc->getName() == "_ZL10cudaMallocIdE9cudaErrorPPT_m"){
          auto devDataPtr = CI->getArgOperand(0);
          auto AllocSize = CI->getArgOperand(1);
          PointerType* Ty = dyn_cast<PointerType>(devDataPtr->getType());

          Instruction* Malloc = CallInst::CreateMalloc(CI,
                                             AllocSize->getType(), Ty->getElementType(), AllocSize,
                                             nullptr, nullptr, "");

          for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
              for (Instruction::op_iterator I_Op = (&*I)->op_begin(), E_Op = (&*I)->op_end(); I_Op != E_Op; ++I_Op){
                if(*I_Op == devDataPtr){
                  *I_Op = Malloc;
                }
              }
          }
          insts2Remove.push_back(CI);
        }
        else if(calledFunc->getName() == "cudaMemcpy"){
          Type* Int1Ty = Type::getInt1Ty(F.getContext());
          CallSite CS(CI);
          SmallVector<Value *, 4> Args(CS.arg_begin(), CS.arg_end()-1);
          Args.push_back(ConstantInt::getFalse(Int1Ty));
          ArrayRef<Value*> args(Args);
          std::vector<Type*> argTyVec;
          argTyVec.push_back(PointerType::get(Type::getInt8Ty(F.getContext()), 0));
          argTyVec.push_back(PointerType::get(Type::getInt8Ty(F.getContext()), 0));
          argTyVec.push_back(Type::getInt64Ty(F.getContext()));
          argTyVec.push_back(Int1Ty);
          ArrayRef<Type *> argTys(argTyVec);
          FunctionType* memcpyFuncTy = FunctionType::get(
              Type::getVoidTy(F.getContext()), //return type
              argTys,
              false
          );

          auto MemCpyFunc = F.getParent()->getOrInsertFunction("llvm.memcpy.p0i8.p0i8.i64", memcpyFuncTy);
          CallInst *NewCI = CallInst::Create(MemCpyFunc, args, "", CI);
          insts2Remove.push_back(CI);
        }
        else if(calledFunc->getName() == "cudaDeviceSynchronize"){
          insts2Remove.push_back(CI);
        }
      }
    }
    for(auto I : insts2Remove)
      I->eraseFromParent();

    return false;
  }
}; // end of struct Hello
}  // end of anonymous namespace

char MergeKernel::ID = 0;
static RegisterPass<MergeKernel> X("merge-kernel", "merge cuda kernel back to main file",
                             false /* Only looks at CFG */,
                             false /* Analysis Pass */);
