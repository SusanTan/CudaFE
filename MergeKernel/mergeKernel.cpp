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
