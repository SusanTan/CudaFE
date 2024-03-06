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
#include "llvm/Analysis/LoopInfo.h"


#include <set>
#include <stack>
#include <map>
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/Operator.h"

//for noelle integration
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <algorithm>
#include "noelle/core/Noelle.hpp"
#include "noelle/core/ReductionSCC.hpp"
#include "noelle/core/BinaryReductionSCC.hpp"
#include "llvm/IR/Metadata.h"
#include "Parallelizer.hpp"

using namespace llvm::noelle;

namespace {
struct NoelleReduction : public ModulePass {
  static char ID;
  NoelleReduction() : ModulePass(ID) {}

  bool doInitialization (Module &M) override {
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<Noelle>();
    AU.addRequired<HeuristicsPass>();
  }

  bool runOnModule(Module &M) override {
      /*
       * Fetch NOELLE & heuristics
       */
      auto& noelle = getAnalysis<Noelle>();
      auto heuristics = getAnalysis<HeuristicsPass>().getHeuristics(noelle);

      /*
       * Fetch the PDG
       */
      auto PDG = noelle.getProgramDependenceGraph();

      /*
       * Allocate the parallelization techniques.
       */
      DOALL doall{ noelle };

      for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
        Function *F = &*FI;
        if(F->isDeclaration()) continue;
        auto FDG = noelle.getFunctionDependenceGraph(F);
        auto &LI = getAnalysis<LoopInfoWrapperPass>(*F).getLoopInfo();
        for (auto l : LI.getLoopsInPreorder()) {
          auto LDG = FDG->createLoopsSubgraph(l);
          auto ls = new LoopStructure(l);
          auto optimizations = {
            LoopDependenceInfoOptimization::MEMORY_CLONING_ID,
            LoopDependenceInfoOptimization::THREAD_SAFE_LIBRARY_ID
          };
          auto LDI = noelle.getLoop(ls, optimizations);
          auto loopPreHeader = ls->getPreHeader();
          auto sccManager = LDI->getSCCManager();
          auto loopSCCDAG = sccManager->getSCCDAG();
          /*
           * Fetch the environment of the loop
           */
          auto environment = LDI->getEnvironment();
          assert(environment != nullptr);
          //SUSAN: logic pulled from noelle/src/tools/parallelization_technique/src/ParallelizationTechnique.cpp
          /*
           * Collect reduction operation information needed to accumulate reducable
           * variables after parallelization execution
           */
          std::unordered_map<uint32_t, Instruction::BinaryOps> reducableBinaryOps;
          std::unordered_map<uint32_t, Value *> initialValues;
          for (auto envID : environment->getEnvIDsOfLiveOutVars()) {

            Instruction *producer = dyn_cast<Instruction>(environment->getProducer(envID));
            auto producerSCC = loopSCCDAG->sccOfValue(producer);
            auto producerSCCAttributes =
                dyn_cast<BinaryReductionSCC>(sccManager->getSCCAttrs(producerSCC));
            if (!producerSCCAttributes)
              continue;
            errs() << "SUSAN??\n";
            auto initialValue = producerSCCAttributes->getInitialValue();
            auto reduceOp = producerSCCAttributes->getReductionOperation();
            LLVMContext& C = producer->getContext();
            MDNode* N = MDNode::get(C, MDString::get(C, ""));
            switch (reduceOp) {
              case Instruction::Add:
              case Instruction::FAdd:
                producer->setMetadata("tulip.reduce.add", N);
                break;
              case Instruction::Sub:
              case Instruction::FSub:
                producer->setMetadata("tulip.reduce.sub", N);
                break;
              case Instruction::Mul:
              case Instruction::FMul:
                producer->setMetadata("tulip.reduce.mul", N);
                break;
              case Instruction::UDiv:
              case Instruction::SDiv:
              case Instruction::FDiv:
                producer->setMetadata("tulip.reduce.div", N);
                break;
              case Instruction::URem:
              case Instruction::SRem:
              case Instruction::FRem:
                producer->setMetadata("tulip.reduce.rem", N);
                break;
              case Instruction::Shl:
                producer->setMetadata("tulip.reduce.shl", N);
                break;
              case Instruction::LShr:
              case Instruction::AShr:
                producer->setMetadata("tulip.reduce.shr", N);
                break;
              case Instruction::And:
                producer->setMetadata("tulip.reduce.and", N);
                break;
              case Instruction::Or:
                producer->setMetadata("tulip.reduce.or", N);
                break;
              case Instruction::Xor:
                producer->setMetadata("tulip.reduce.xor", N);
                break;
            }
          }


          // Apply DOALL
          auto ltm = LDI->getLoopTransformationsManager();
          if (true && noelle.isTransformationEnabled(DOALL_ID)
              && ltm->isTransformationEnabled(DOALL_ID)
              && doall.canBeAppliedToLoop(LDI, heuristics)) {
            errs() << "SUSAN: DOALL can be applied\n";
          }
        }
      }

      return false;
  }


}; // end of struct Hello
}  // end of anonymous namespace

char NoelleReduction::ID = 0;
static RegisterPass<NoelleReduction> X("noelle-reduction", "Use noelle reduction pass to mark reduction objections",
                             false /* Only looks at CFG */,
                             false /* Analysis Pass */);
