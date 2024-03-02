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
  }

  bool runOnModule(Module &M) override {
      /*
       * Fetch NOELLE
       */
      auto& noelle = getAnalysis<Noelle>();

      /*
       * Fetch the PDG
       */
      auto PDG = noelle.getProgramDependenceGraph();


      for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
        Function *F = &*FI;
        if(F->isDeclaration()) continue;
        auto FDG = noelle.getFunctionDependenceGraph(F);
        auto &LI = getAnalysis<LoopInfoWrapperPass>(*F).getLoopInfo();
        for (auto l : LI.getLoopsInPreorder()) {
          auto LDG = FDG->createLoopsSubgraph(l);
          auto ls = new LoopStructure(l);
          auto LDI = noelle.getLoop(ls);
          auto sccManager = LDI->getSCCManager();
          auto sccdag = new SCCDAG(FDG);
          for (auto sccNode : sccdag->getNodes()) {
            auto scc = sccNode->getT();
            auto sccInfo = sccManager->getSCCAttrs(scc);
            if(sccInfo){
              errs() << "SUSAN: reduction\n";
              scc->print(errs());
            }
            if (sccInfo && isa<ReductionSCC>(sccInfo)){
              errs() << "SUSAN: found reduction scc\n";
              scc->print(errs());
            }
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
