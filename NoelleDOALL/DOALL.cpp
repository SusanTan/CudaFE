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
struct NoelleDOALL : public ModulePass {
  static char ID;
  NoelleDOALL() : ModulePass(ID) {}

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
       * Fetch NOELLE & parallelization related things
       */
      auto& noelle = getAnalysis<Noelle>();
      auto heuristics = getAnalysis<HeuristicsPass>().getHeuristics(noelle);
      auto mm = noelle.getMetadataManager();
      auto forest = noelle.getLoopNestingForest();
      if (forest->getNumberOfLoops() == 0) {
        errs() << "Parallelizer:    There is no loop to consider\n";

        /*
         * Free the memory.
         */
        delete forest;

        errs() << "Parallelizer: Exit\n";
        return false;
      }

      /*
       * Fetch the PDG
       */
      auto PDG = noelle.getProgramDependenceGraph();


      int parallelizationOrderIndex = 0;
      for (auto tree : forest->getTrees()) {
          /*order the loop*/
          std::map<LoopStructure *, bool> doallLoops;
          std::vector<LoopDependenceInfo *> selectedLoops{};
          auto selector = [&noelle,
                           &doallLoops,
                           &selectedLoops](LoopForestNode *n, uint32_t treeLevel) -> bool {
            /*
             * Tag DOALL loops.
             */
             auto ls = n->getLoop();
             auto optimizations = {
               LoopDependenceInfoOptimization::MEMORY_CLONING_ID,
               LoopDependenceInfoOptimization::THREAD_SAFE_LIBRARY_ID
             };
             auto ldi = noelle.getLoop(ls, optimizations);
             doallLoops[ls] = true;
             selectedLoops.push_back(ldi);
            return false;
          };
          tree->visitPreOrder(selector);


        auto loopsToParallelize = selectedLoops;
        for (auto ldi : loopsToParallelize) {
          auto ls = ldi->getLoopStructure();
          auto ldiParallelizationOrderIndex =
              std::to_string(parallelizationOrderIndex++);
          mm->addMetadata(ls,
                          "noelle.parallelizer.looporder",
                          ldiParallelizationOrderIndex);
        }
      }
      /*
       * Allocate the parallelization techniques.
       */
      DOALL doall{ noelle };

      std::map<uint32_t, LoopDependenceInfo *> loopParallelizationOrder;
      for (auto tree : forest->getTrees()) {
        auto selector = [&noelle, &mm, &loopParallelizationOrder](
                            LoopForestNode *n,
                            uint32_t treeLevel) -> bool {
          auto ls = n->getLoop();
          if (!mm->doesHaveMetadata(ls, "noelle.parallelizer.looporder")) {
            return false;
          }
          auto parallelizationOrderIndex =
              std::stoi(mm->getMetadata(ls, "noelle.parallelizer.looporder"));
          auto optimizations = {
            LoopDependenceInfoOptimization::MEMORY_CLONING_ID,
            LoopDependenceInfoOptimization::THREAD_SAFE_LIBRARY_ID
          };
          auto ldi = noelle.getLoop(ls, optimizations);
          loopParallelizationOrder[parallelizationOrderIndex] = ldi;
          return false;
        };
        tree->visitPreOrder(selector);
      }

      /*
       * Parallelize the loops in order.
       */
      std::unordered_map<BasicBlock *, bool> modifiedBBs{};
      uint32_t parallelizedIndex = 0;
      for (auto indexLoopPair : loopParallelizationOrder) {
        errs() << "SUSAN???\n";
        auto ldi = indexLoopPair.second;
        auto ls = ldi->getLoopStructure();
        auto loopIDOpt = ls->getID();
        auto environment = ldi->getEnvironment();
        auto optimizations = {
          LoopDependenceInfoOptimization::MEMORY_CLONING_ID,
          LoopDependenceInfoOptimization::THREAD_SAFE_LIBRARY_ID
        };
        auto loopPreHeader = ls->getPreHeader();
        auto sccManager = ldi->getSCCManager();
        auto loopSCCDAG = sccManager->getSCCDAG();
        for (auto envID : environment->getEnvIDsOfLiveOutVars()) {
            Instruction *producer = dyn_cast<Instruction>(environment->getProducer(envID));
            auto producerSCC = loopSCCDAG->sccOfValue(producer);
            auto producerSCCAttributes =
                dyn_cast<BinaryReductionSCC>(sccManager->getSCCAttrs(producerSCC));
            if (!producerSCCAttributes)
              continue;
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
        /*
         * Check if we can parallelize this loop.
         */
        auto safe = true;
        for (auto bb : ls->getBasicBlocks()) {
          if (modifiedBBs[bb]) {
            safe = false;
            break;
          }
        }
        if (!safe) {
          errs() << "Parallelizer:    Loop ";
          // Parent loop has been parallelized, so basic blocks have been modified
          // and we might not have a loop ID for the child loop. If we have it we
          // print it, otherwise we don't.
          if (loopIDOpt) {
            auto loopID = loopIDOpt.value();
            errs() << loopID;
          }
          errs()
              << " cannot be parallelized because one of its parent has been parallelized already\n";
          continue;
        }

        // Apply DOALL
        auto ltm = ldi->getLoopTransformationsManager();
        if (true && noelle.isTransformationEnabled(DOALL_ID)
            && ltm->isTransformationEnabled(DOALL_ID)
            && doall.canBeAppliedToLoop(ldi, heuristics)) {
          mm->addMetadata(ls, "noelle.doall.loop", std::to_string(0));
          for (auto bb : ls->getBasicBlocks()) {
            modifiedBBs[bb] = true;
          }
        }
      }

      return false;
  }


}; // end of struct Hello
}  // end of anonymous namespace

char NoelleDOALL::ID = 0;
static RegisterPass<NoelleDOALL> X("noelle-doall", "Use noelle reduction pass to mark reduction objections",
                             false /* Only looks at CFG */,
                             false /* Analysis Pass */);
