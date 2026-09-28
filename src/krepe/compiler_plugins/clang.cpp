// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/Basic/Version.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendPluginRegistry.h>
#include <clang/Sema/Sema.h>
using namespace clang;

#if (__clang_major__ != CLANG_VERSION_MAJOR || \
     __clang_minor__ != CLANG_VERSION_MINOR)
static_assert(false,
              "Major or minor version mismatch between the compiler version "
              "and the plugin headers version (" __clang_version__
              " vs " CLANG_VERSION_STRING ")");
#endif

namespace {

ImplicitCastExpr* get_function_ptr_expr(const ASTContext& ast,
                                        FunctionDecl* function) {
  DeclRefExpr* function_ref_expr = DeclRefExpr::Create(
      ast, NestedNameSpecifierLoc{}, SourceLocation{}, function, false,
      SourceLocation{}, function->getType(), VK_LValue);

  return ImplicitCastExpr::Create(ast, ast.getPointerType(function->getType()),
                                  CK_FunctionToPointerDecay, function_ref_expr,
                                  nullptr, VK_PRValue, FPOptionsOverride{});
}

CallExpr* build_function_call(const CompilerInstance& compiler,
                              FunctionDecl* function, ArrayRef<Expr*> args,
                              ImplicitCastExpr* function_ptr = nullptr) {
  auto& ast  = compiler.getASTContext();
  auto& sema = compiler.getSema();

  if (!function_ptr) {
    function_ptr = get_function_ptr_expr(ast, function);
  }

  sema.MarkFunctionReferenced(SourceLocation{}, function);

  return CallExpr::Create(ast, function_ptr, args, function->getReturnType(),
                          VK_PRValue, SourceLocation{}, FPOptionsOverride{});
}

QualType get_canonical_type(const QualType& type) {
  return type.getNonReferenceType().getUnqualifiedType().getCanonicalType();
}

CXXRecordDecl* get_struct_decl(const QualType& type) {
  return get_canonical_type(type)->getAsCXXRecordDecl();
}

template <class Predicate>
NamedDecl* get_enclosed_decl(const DeclContext* decl, const char* name,
                             ASTContext& ast, Predicate p) {
  IdentifierInfo& id = ast.Idents.get(name);
  DeclarationName decl_name(&id);
  DeclContextLookupResult results = decl->lookup(decl_name);

  auto it = std::find_if(results.begin(), results.end(), p);

  if (it == results.end()) {
    return nullptr;
  }

  return *it;
}

struct ViewInfo {
  // MemberExpr for view.data
  MemberExpr* data_method = nullptr;
  // Decl node for the data() method
  CXXMethodDecl* data_method_decl = nullptr;
  // Decl node for the memory space type
  CXXRecordDecl* memory_space = nullptr;
  // Decl node for the memory space's name() function
  CXXMethodDecl* memory_space_name_function = nullptr;
};

class FunctorVisitor {
  ASTContext& ast;
  std::vector<ViewInfo> views;

 public:
  FunctorVisitor(ASTContext& ast) : ast(ast) {}

  const std::vector<ViewInfo>& get_views() const { return views; }

  bool visit_view(const CXXRecordDecl* view, Expr* root) {
    if (view->getName() != "View") {
      return false;
    }

    const NamespaceDecl* enclosing_namespace =
        dyn_cast<NamespaceDecl>(view->getEnclosingNamespaceContext());
    if (!enclosing_namespace || enclosing_namespace->isAnonymousNamespace() ||
        enclosing_namespace->getName() != "Kokkos") {
      return false;
    }

    auto data_method = dyn_cast<CXXMethodDecl>(
        get_enclosed_decl(view, "data", ast, [](NamedDecl* decl) {
          CXXMethodDecl* method = dyn_cast<CXXMethodDecl>(decl);
          return method && method->param_size() == 0;
        }));

    if (!data_method) {
      llvm::errs() << "Failed to find the data method of a View\n";
      abort();
    }

    MemberExpr* data_member_expr = MemberExpr::CreateImplicit(
        ast, root, false, data_method, data_method->getType(), VK_LValue,
        OK_Ordinary);

    // We search for TypedefNameDecl in order to cover both using and typedef
    auto memory_space_alias = dyn_cast<TypedefNameDecl>(
        get_enclosed_decl(view, "memory_space", ast, [](NamedDecl* decl) {
          return dyn_cast<TypedefNameDecl>(decl) != nullptr;
        }));

    if (!memory_space_alias) {
      llvm::errs() << "Failed to find the memory_space attribute of a View\n";
      abort();
    }

    CXXRecordDecl* memory_space =
        get_struct_decl(memory_space_alias->getUnderlyingType());

    auto memory_space_name_function = dyn_cast<CXXMethodDecl>(
        get_enclosed_decl(memory_space, "name", ast, [](NamedDecl* decl) {
          CXXMethodDecl* method = dyn_cast<CXXMethodDecl>(decl);
          return method && method->isStatic() && method->param_size() == 0;
        }));

    if (!memory_space_name_function) {
      llvm::errs() << "Failed to find the name function of a memory space\n";
      abort();
    }

    views.push_back({data_member_expr, data_method, memory_space,
                     memory_space_name_function});

    return true;
  }

  void visit(const FieldDecl* field, Expr* root) {
    CXXRecordDecl* struct_decl = get_struct_decl(field->getType());
    if (!struct_decl) {
      return;
    }

    if (!visit_view(struct_decl, root)) {
      visit(struct_decl, root);
    }
  }

  void visit(CXXRecordDecl* record, Expr* root) {
    for (FieldDecl* field : record->fields()) {
      MemberExpr* new_root = MemberExpr::CreateImplicit(
          ast, root, false, field, field->getType(), VK_LValue, OK_Ordinary);
      visit(field, new_root);
    }
  }
};

class RegisterFunctorViewsConsumer : public ASTConsumer {
  CompilerInstance& compiler;

  void patch_replay_functor(FunctionDecl* fun) {
    const auto params = fun->parameters();
    if (params.size() != 1) {
      llvm::errs() << "krepe::replay_functor should have 1 parameter, got "
                   << params.size() << '\n';
      abort();
    }

    ParmVarDecl* functor          = params[0];
    CXXRecordDecl* functor_struct = get_struct_decl(functor->getType());
    if (!functor_struct) {
      llvm::errs()
          << "The argument to krepe::replay_functor should be a record type\n";
      abort();
    }

    ASTContext& ast   = compiler.getASTContext();
    DeclRefExpr* root = DeclRefExpr::Create(
        ast, NestedNameSpecifierLoc{}, SourceLocation{},
        dyn_cast<VarDecl>(functor), false, functor->getLocation(),
        functor->getType(), VK_LValue);
    if (!root) {
      llvm::errs() << "Failed to create root RefExpr\n";
      abort();
    }

    FunctorVisitor functor_visitor(ast);
    functor_visitor.visit(functor_struct, root);
    const std::vector<ViewInfo>& views = functor_visitor.get_views();

    const NamespaceDecl* krepe_namespace =
        dyn_cast<NamespaceDecl>(fun->getEnclosingNamespaceContext());

    const NamespaceDecl* impl_namespace = dyn_cast<NamespaceDecl>(
        get_enclosed_decl(krepe_namespace, "impl", ast, [](NamedDecl* decl) {
          return dyn_cast<NamespaceDecl>(decl) != nullptr;
        }));

    if (!impl_namespace) {
      llvm::errs() << "Failed to find the krepe::impl namespace\n";
      abort();
    }

    FunctionDecl* clear_registered_views_function =
        dyn_cast<FunctionDecl>(get_enclosed_decl(
            impl_namespace, "clear_registered_views", ast, [](NamedDecl* decl) {
              FunctionDecl* func = dyn_cast<FunctionDecl>(decl);
              return func && func->param_size() == 0;
            }));

    if (!clear_registered_views_function) {
      llvm::errs() << "Failed to find the krepe::impl::clear_registered_views "
                      "function\n";
      abort();
    }

    FunctionDecl* register_view_function =
        dyn_cast<FunctionDecl>(get_enclosed_decl(
            impl_namespace, "register_view", ast, [](NamedDecl* decl) {
              FunctionDecl* func = dyn_cast<FunctionDecl>(decl);
              return func && func->param_size() == 2;
            }));

    if (!register_view_function) {
      llvm::errs()
          << "Failed to find the krepe::impl::register_view function\n";
      abort();
    }

    auto register_view_ptr = get_function_ptr_expr(ast, register_view_function);

    std::vector<Stmt*> new_body_stmts;
    new_body_stmts.reserve(views.size() + 2);

    auto clear_registered_views_call =
        build_function_call(compiler, clear_registered_views_function, {});
    new_body_stmts.push_back(clear_registered_views_call);

    Sema& sema = compiler.getSema();

    for (auto& [data_method_expr, data_method_decl, memory_space,
                name_function_decl] : views) {
      sema.MarkFunctionReferenced(SourceLocation{}, data_method_decl);

      auto data_call = CXXMemberCallExpr::Create(
          ast, data_method_expr, {}, data_method_decl->getReturnType(),
          VK_PRValue, SourceLocation{}, FPOptionsOverride{});

      auto data_ptr =
          ImplicitCastExpr::Create(ast, ast.VoidPtrTy, CK_BitCast, data_call,
                                   nullptr, VK_PRValue, FPOptionsOverride{});

      auto name_call = build_function_call(compiler, name_function_decl, {});

      std::array<Expr*, 2> register_view_args = {
          data_ptr,
          name_call,
      };
      auto register_view_call =
          build_function_call(compiler, register_view_function,
                              register_view_args, register_view_ptr);

      new_body_stmts.push_back(register_view_call);
    }

    Stmt* body = fun->getBody();
    new_body_stmts.push_back(body);

    auto new_body =
        CompoundStmt::Create(ast, new_body_stmts, FPOptionsOverride{},
                             fun->getBeginLoc(), fun->getBodyRBrace());

    fun->setBody(new_body);
  }

 public:
  RegisterFunctorViewsConsumer(CompilerInstance& Instance)
      : compiler(Instance) {}

  bool HandleTopLevelDecl(DeclGroupRef group) override {
    for (DeclGroupRef::iterator it = group.begin(), end = group.end();
         it != end; ++it) {
      Decl* decl             = *it;
      FunctionDecl* function = dyn_cast<FunctionDecl>(decl);
      if (!function || !function->getIdentifier() ||
          function->getName() != "replay_functor") {
        continue;
      }

      const NamespaceDecl* krepe_namespace =
          dyn_cast<NamespaceDecl>(function->getEnclosingNamespaceContext());
      if (!krepe_namespace || krepe_namespace->isAnonymousNamespace() ||
          krepe_namespace->getName() != "krepe") {
        continue;
      }

      patch_replay_functor(function);
    }

    return ASTConsumer::HandleTopLevelDecl(group);
  }
};

class RegisterFunctorViewsAction : public PluginASTAction {
 protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& compiler,
                                                 llvm::StringRef) override {
    return std::make_unique<RegisterFunctorViewsConsumer>(compiler);
  }

  bool ParseArgs(const CompilerInstance&,
                 const std::vector<std::string>&) override {
    return true;
  }

  PluginASTAction::ActionType getActionType() override {
    return AddBeforeMainAction;
  }
};

}  // namespace

static FrontendPluginRegistry::Add<RegisterFunctorViewsAction> X(
    "register_views",
    "Register views used in functors passed to krepe::replay_functor");
