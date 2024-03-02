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
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/Operator.h"

//for noelle integration
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <algorithm>

#include "noelle/core/Noelle.hpp"
using namespace llvm::noelle;

namespace {
struct NoelleReduction : public ModulePass {
  static char ID;
  NoelleReduction() : ModulePass(ID) {}

  bool doInitialization (Module &M) override {
    return false;
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

      /*
       * Fetch the FDG of "main"
       */
      auto fm = noelle.getFunctionsManager();
      auto mainF = fm->getEntryFunction();
      auto FDG = noelle.getFunctionDependenceGraph(mainF);

      /*
       * Compute the SCCDAG of the FDG of "main"
       */
      auto mainSCCDAG = new SCCDAG(FDG);

      return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<Noelle>();
  }
}; // end of struct Hello
}  // end of anonymous namespace

char NoelleReduction::ID = 0;
static RegisterPass<NoelleReduction> X("noelle-reduction", "Use noelle reduction pass to mark reduction objections",
                             false /* Only looks at CFG */,
                             false /* Analysis Pass */);
