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

using namespace llvm;

namespace {
struct MergeKernel : public FunctionPass {
  static char ID;
  std::vector<Value*> loopDims;
  MergeKernel() : FunctionPass(ID) {}

  void findThreadDim(Function &F, LoadInst *DimArg){
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
      if(CI->getCalledFunction()->getName() != "_ZN4dim3C2Ejjj") continue;
      if(CI->getArgOperand(0) != AggAlloca) continue;

      for(int i=1; i<=3; ++i){
        Value *dim = CI->getArgOperand(i);
        if(ConstantInt *dimConst = dyn_cast<ConstantInt>(dim))
          if(dimConst->getZExtValue() == 1)
            continue;
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
        if(CI->getCalledFunction()->getName() != "_ZN4dim3C2Ejjj") continue;
        if(CI->getArgOperand(0) != DimAlloca) continue;
        for(int i=1; i<=3; ++i){
          Value *dim = CI->getArgOperand(i);
          if(ConstantInt *dimConst = dyn_cast<ConstantInt>(dim))
            if(dimConst->getZExtValue() == 1)
              continue;
          loopDims.push_back(dim);
          errs() << "mergeKernel: Dim " << i << " : " << *dim <<"\n";
        }
        foundDim = true;
        break;
      }
    }
  }

  bool runOnFunction(Function &F) override {
    loopDims.clear();
    std::vector<Instruction*> insts2Remove;
    BasicBlock *kernelBB = nullptr;
    for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
      if(CallInst *CI = dyn_cast<CallInst>(&*I)){
        Function* calledFunc = CI->getCalledFunction();
        if(calledFunc->getName() == "_ZL10cudaMallocIdE9cudaErrorPPT_m" ||
           calledFunc->getName() == "_ZN4dim3C2Ejjj"){
          insts2Remove.push_back(CI);
        }
        else if(calledFunc->getName() == "cudaConfigureCall"){
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
                insts2Remove.push_back(cmp);
              }
            }
          }

          errs() << "MergeKernel: blocks per grid:\n";
          findThreadDim(F, dyn_cast<LoadInst>(CI->getArgOperand(0)));
          errs() << "MergeKernel: threads per block:\n";
          findThreadDim(F, dyn_cast<LoadInst>(CI->getArgOperand(2)));

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
        auto header = BasicBlock::Create(kernelBB->getContext(), "header." + std::to_string(loopCnt), &F, kernelBB);
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
        auto latch = BasicBlock::Create(kernelBB->getContext(), "latch." + std::to_string(i), &F, kernelBB);
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
        CmpInst *cmp = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_ULT, indvar, header2itNum[header], "exitCheck." + std::to_string(i), header);
        BranchInst *term = nullptr;

        //create increment in latch
        auto latch = header2latch[header];
        Value *incr = BinaryOperator::Create(Instruction::Add, indvar, ConstantInt::get(phiTy, 1),
                                  "indvar.next." + std::to_string(i), latch->getTerminator());

        if(!prevHeader){
          term = BranchInst::Create(kernelBB, header2latch[nextHeader], cmp, header);
          indvar->addIncoming(ConstantInt::get(phiTy,0), nextHeader);
        }
        else if(!nextHeader){
          term = BranchInst::Create(prevHeader, loopExit, cmp, header);
          indvar->addIncoming(ConstantInt::get(phiTy,0), kernelPred);
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

    //delete cuda calls and control flows
    for(auto I : insts2Remove)
      I->eraseFromParent();

    errs() << F << "\n";
    return false;
  }
}; // end of struct Hello
}  // end of anonymous namespace

char MergeKernel::ID = 0;
static RegisterPass<MergeKernel> X("merge-kernel", "merge cuda kernel back to main file",
                             false /* Only looks at CFG */,
                             false /* Analysis Pass */);
