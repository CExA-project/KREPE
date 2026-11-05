#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <clang/Basic/Version.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendPluginRegistry.h>
#include <clang/Sema/Sema.h>

#include <algorithm>
#include <filesystem>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace clang;

#if (__clang_major__ != CLANG_VERSION_MAJOR || \
     __clang_minor__ != CLANG_VERSION_MINOR)
static_assert(false,
              "Major or minor version mismatch between the compiler version "
              "and the plugin headers version (" __clang_version__
              " vs " CLANG_VERSION_STRING ")");
#endif

#define CLANG_VERSION_LESS(MAJOR, MINOR, PATCH)                        \
  (CLANG_VERSION_MAJOR < MAJOR) ||                                     \
      (CLANG_VERSION_MAJOR == MAJOR && CLANG_VERSION_MINOR < MINOR) || \
      (CLANG_VERSION_MAJOR == MAJOR && CLANG_VERSION_MINOR == MINOR && \
       CLANG_VERSION_PATCH < PATCH)

#define CLANG_VERSION_GREATER_EQUAL(MAJOR, MINOR, PATCH) \
  (!CLANG_VERSION_LESS(MAJOR, MINOR, PATCH))

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

/**
 * @brief Finds the LambdaExpr corresponding to the Functor class of a
 * parallel_for instantiation, as well as the CallExpr that led to this
 * instantiation.
 *
 * NOTE: for the CallExpr, this kind of assumes that there is only one call
 * using this lambda type. This shouldn't be an important issue since it's
 * mostly used to get a better name for the policy variable.
 */
class FindLambdaExprVisitor
    : public RecursiveASTVisitor<FindLambdaExprVisitor> {
 private:
  const CXXRecordDecl* target_record;
  const FunctionDecl* target_fun;
  LambdaExpr* lambda_expr = nullptr;
  CallExpr* function_call = nullptr;

 public:
  FindLambdaExprVisitor(const CXXRecordDecl* target,
                        const FunctionDecl* target_fun)
      : target_record(target), target_fun(target_fun) {}

  std::pair<LambdaExpr*, CallExpr*> get_results() {
    return {lambda_expr, function_call};
  }

  bool VisitLambdaExpr(LambdaExpr* lambda) {
    if (!lambda_expr && lambda->getLambdaClass() == target_record) {
      lambda_expr = lambda;
    }
    return true;
  }

  bool VisitCallExpr(CallExpr* call) {
    if (const FunctionDecl* fun = call->getDirectCallee();
        fun && fun == target_fun) {
      function_call = call;
    }
    return true;
  }
};

/**
 * @brief Visitor used to traverse the AST starting from a LambdaExpr or
 * CXXRecordDecl corresponding to the Functor template argument of a
 * krepe::parallel_for call in order to collect the declarations the functor
 * depends on.
 */
class FindUsedDeclsVisitor : public RecursiveASTVisitor<FindUsedDeclsVisitor> {
 private:
  CompilerInstance& compiler;
  ASTContext& ast;
  SourceManager& source_manager;
  Decl* excluded_fun = nullptr;
  std::unordered_set<std::string> includes;
  const std::vector<std::string>& include_dirs;
  std::unordered_set<const Decl*> visited_decls;
  std::unordered_set<const Decl*> used_decls;

  /**
   * @brief Get a name suitable to #include `filename`
   *
   * @param filename The file to include
   * @return The relative filename to use in #include if `filename` can be
   * reached through the include directories, or nullopt otherwise
   */
  std::optional<std::string> get_include_name(const std::string& filename) {
    for (const auto& path : include_dirs) {
      if (!filename.starts_with(path)) {
        continue;
      }

      return filename.substr(path.size() + 1);
    }

    return std::nullopt;
  }

  /**
   * @brief Get the relative filename to #include an entity from a specific
   * SourceLocation
   *
   * @param loc The SourceLocation of the entity we want to include
   * @param is_in_std_namespace Whether this entity resides in the std namespace
   *        or a namespace with a reserved name
   * @return The relative filename to use in #include if `loc` can be
   * reached through the include directories, or nullopt otherwise
   */
  std::optional<std::string> get_include_name(SourceLocation loc,
                                              bool is_in_std_namespace) {
    SourceLocation includeLoc = loc;
    PresumedLoc prev;

    auto is_std_header =
        [](std::string_view path) -> std::optional<std::string_view> {
      std::string_view header_name = path.substr(path.find_last_of('/') + 1);
      if (std::all_of(header_name.begin(), header_name.end(), [](char c) {
            return ('a' <= c && c <= 'z') || c == '_';
          })) {
        return header_name;
      }
      return std::nullopt;
    };

    while (!source_manager.isInMainFile(includeLoc)) {
      PresumedLoc PLoc = source_manager.getPresumedLoc(includeLoc);
      if (is_in_std_namespace) {
        if (auto std_header = is_std_header(PLoc.getFilename())) {
          return std::string(std_header.value());
        }
      }
      FileID parentFid = source_manager.getFileID(includeLoc);
      includeLoc       = source_manager.getIncludeLoc(parentFid);
      prev             = PLoc;
    }

    if (prev.isInvalid()) {
      return std::nullopt;
    }

    return get_include_name(prev.getFilename());
  }

  /**
   * @brief Tells if a DeclContext is or is contained inside the std namespace
   * or a namespace with a reserved identifier
   *
   * Using isInStdNamespace is not always reliable, it fails with inline
   * namespaces and doesn't apply to entities in internal standard library
   * impementations namespaces.
   *
   * We don't use heuristics based on whether the decl appears inside a system
   * headers since regular libraries can also be included as system headers.
   *
   * @param context The DeclContext to check
   * @return true if `context` is or lives under the std or a reserved namespace
   */
  static bool is_standard_library_decl(const DeclContext* context) {
    if (!isa<NamespaceDecl>(*context)) {
      context = context->getEnclosingNamespaceContext();
    }

    const NamespaceDecl* prev = nullptr;
    const NamespaceDecl* ns   = dyn_cast<NamespaceDecl>(context);
    while (ns && ns != prev) {
      if (!ns->isAnonymousNamespace() && ns->getName().starts_with("__")) {
        return true;
      }
      prev = ns;
      ns   = dyn_cast<NamespaceDecl>(ns->getEnclosingNamespaceContext());
    }
    return ns && !ns->isAnonymousNamespace() && ns->getName() == "std";
  }

  /**
   * @brief Determines whether an entity from a specific SourceLocation can be
   * #included
   *
   * If the entity can be included, the function also stores the name to use
   * inside the brackets of an `#include <>` directive.
   *
   * @param location The SourceLocation of the entity to #include
   * @param context The enclosing DeclContext of the entity
   * @param is_in_std_namespace The result of calling `isInStdNamespace()` on
   *   	    the entity's type definition
   * @return true if the entity wan be #included, false otherwise
   */
  bool try_include_decl(const SourceLocation location,
                        const DeclContext* context, bool is_in_std_namespace) {
    SourceLocation loc = source_manager.getExpansionLoc(location);

    if (source_manager.isInMainFile(loc)) {
      return false;
    }

    std::optional<std::string> include_filename = get_include_name(
        loc, is_in_std_namespace || is_standard_library_decl(context));
    if (!include_filename) {
      return false;
    }

    includes.insert(include_filename.value());
    return true;
  }

  /**
   * @brief Determines whether a Decl can be #included, see the other overload
   * for details
   */
  bool try_include_decl(const Decl* decl) {
    return try_include_decl(decl->getLocation(), decl->getDeclContext(),
                            decl->isInStdNamespace());
  }

 public:
  FindUsedDeclsVisitor(CompilerInstance& compiler,
                       const std::vector<std::string>& include_dirs)
      : compiler(compiler),
        ast(compiler.getASTContext()),
        source_manager(ast.getSourceManager()),
        include_dirs(include_dirs) {}

  std::unordered_set<const Decl*>& get_used_decls() { return used_decls; }
  std::unordered_set<std::string>& get_includes() { return includes; }

  void set_excluded_function(Decl* fun) { excluded_fun = fun; }

  bool shouldVisitLambdaBody() const { return true; }

  bool VisitLambdaExpr(LambdaExpr* expr) {
    if (try_include_decl(expr->getBeginLoc(),
                         expr->getLambdaClass()->getDeclContext(),
                         expr->getLambdaClass()->isInStdNamespace())) {
      return false;
    }
    return true;
  }

  bool VisitCXXRecordDecl(CXXRecordDecl* decl) {
    for (CXXBaseSpecifier base : decl->bases()) {
      TraverseType(base.getType());
    }

    for (FieldDecl* field : decl->fields()) {
      TraverseDecl(field);
    }

    for (CXXConstructorDecl* ctor : decl->ctors()) {
      TraverseDecl(ctor);
    }

    return false;
  }

  bool VisitDecl(Decl* decl) {
    if (visited_decls.contains(decl)) {
      return false;
    }
    visited_decls.insert(decl);

    if (try_include_decl(decl) || decl == excluded_fun) {
      return false;
    }

    used_decls.insert(decl);

    DeclContext* context = decl->getLexicalDeclContext();
    if (!context->isTranslationUnit()) {
      TraverseDecl(Decl::castFromDeclContext(context));
    }

    return true;
  }

  bool VisitFunctionDecl(FunctionDecl* fun) {
    if (!fun->isFirstDecl()) {
      TraverseDecl(fun->getPreviousDecl());
    }
    if (FunctionDecl* def = fun->getDefinition()) {
      TraverseDecl(def);
    }
    return true;
  }

  bool VisitTypedefNameDecl(TypedefNameDecl* decl) {
    TraverseType(decl->getUnderlyingType());
    return true;
  }

  bool VisitCXXDeductionGuideDecl(CXXDeductionGuideDecl* decl) {
    TraverseDecl(decl->getDeducedTemplate());
    return true;
  }

  bool VisitVarTemplateSpecializationDecl(VarTemplateSpecializationDecl* decl) {
    TraverseDecl(decl->getSpecializedTemplate());
    return true;
  }

  // Depending on the clang version, NestedNameSpecifiers may be passed by value
  // or by pointer
  bool VisitNestedNameSpecifier(NestedNameSpecifier spec) {
    return VisitNestedNameSpecifier(&spec);
  }

  bool VisitNestedNameSpecifier(NestedNameSpecifier* spec) {
#if CLANG_VERSION_LESS(22, 1, 0)
    switch (spec->getKind()) {
      case NestedNameSpecifier::Namespace:
        TraverseDecl(spec->getAsNamespace());
        break;
      case NestedNameSpecifier::NamespaceAlias:
        TraverseDecl(spec->getAsNamespaceAlias());
        break;
#if CLANG_VERSION_LESS(21, 1, 0)
      case NestedNameSpecifier::TypeSpecWithTemplate:
#endif
      case NestedNameSpecifier::TypeSpec:
        TraverseType(QualType(spec->getAsType(), 0));
        break;
      case NestedNameSpecifier::Identifier:
      case NestedNameSpecifier::Global:
      case NestedNameSpecifier::Super: break;
    }
    if (NestedNameSpecifier* prefix = spec->getPrefix()) {
      VisitNestedNameSpecifier(prefix);
    }
#else
    switch (spec->getKind()) {
      case NestedNameSpecifier::Kind::Namespace: {
        auto ns = spec->getAsNamespaceAndPrefix();
        TraverseDecl(const_cast<NamespaceBaseDecl*>(ns.Namespace));
        VisitNestedNameSpecifier(ns.Prefix);
      } break;
      case NestedNameSpecifier::Kind::Type:
        TraverseType(QualType(spec->getAsType(), 0));
        break;
      case NestedNameSpecifier::Kind::Global:
      case NestedNameSpecifier::Kind::Null:
      case NestedNameSpecifier::Kind::MicrosoftSuper: break;
    }
#endif
    return true;
  }

  bool VisitNamespaceDecl(NamespaceDecl* decl) {
    // We don't wont to recurse into the namespace's declarations
    return false;
  }

  bool VisitNamespaceAliasDecl(NamespaceAliasDecl* decl) {
    TraverseDecl(decl->getAliasedNamespace());
    return true;
  }

  bool VisitUsingDirectiveDecl(UsingDirectiveDecl* decl) {
    TraverseDecl(decl->getNominatedNamespaceAsWritten());
    return true;
  }

  bool VisitDecltypeType(DecltypeType* type) {
    TraverseStmt(type->getUnderlyingExpr());
    return true;
  }

  bool VisitValueDecl(ValueDecl* decl) {
    TraverseType(decl->getType());
    return true;
  }

#if CLANG_VERSION_LESS(22, 1, 0)
  bool VisitElaboratedType(ElaboratedType* type) {
    if (NestedNameSpecifier* spec = type->getQualifier()) {
      VisitNestedNameSpecifier(spec);
    }
    return true;
  }
#endif

  bool VisitEnumType(EnumType* type) {
    TraverseDecl(type->getDecl());
    return true;
  }

  bool VisitRecordType(RecordType* type) {
    VisitDecl(type->getDecl());
    // TraverseDecl(type->getDecl());
    return true;
  }
  bool VisitTypedefType(TypedefType* type) {
#if CLANG_VERSION_GREATER_EQUAL(22, 1, 0)
    if (auto qualifier = type->getQualifier()) {
      VisitNestedNameSpecifier(qualifier);
    }
#endif
    TraverseDecl(type->getDecl());
    return true;
  }

  bool VisitTemplateSpecializationType(TemplateSpecializationType* type) {
    if (auto* decl = type->getTemplateName().getAsTemplateDecl()) {
      // FIXME: See if we should visit record templates in a specific manner
      TraverseDecl(dyn_cast<Decl>(decl));
    }
    return true;
  }

  bool VisitUsingShadowDecl(UsingShadowDecl* decl) {
    TraverseDecl(decl->getIntroducer());
    return true;
  }

  bool VisitUsingDecl(UsingDecl* decl) {
    auto spec = decl->getQualifier();
    VisitNestedNameSpecifier(spec);

    DeclContextLookupResult result;
#if CLANG_VERSION_LESS(22, 1, 0)
    switch (spec->getKind()) {
      case NestedNameSpecifier::Namespace:
        result = spec->getAsNamespace()->lookup(decl->getDeclName());
        break;
      case NestedNameSpecifier::NamespaceAlias:
        result = spec->getAsNamespaceAlias()->getNamespace()->lookup(
            decl->getDeclName());
        break;
#if CLANG_VERSION_LESS(21, 1, 0)
      case NestedNameSpecifier::TypeSpecWithTemplate:
#endif
      case NestedNameSpecifier::TypeSpec:
        result = spec->getAsRecordDecl()->lookup(decl->getDeclName());
        break;
      case NestedNameSpecifier::Global:
        result = ast.getTranslationUnitDecl()->lookup(decl->getDeclName());
        break;
      case NestedNameSpecifier::Identifier:
      case NestedNameSpecifier::Super: break;
    }
#else
    switch (spec.getKind()) {
      case NestedNameSpecifier::Kind::Namespace: {
        const NamespaceBaseDecl* ns = spec.getAsNamespaceAndPrefix().Namespace;
        if (isa<NamespaceDecl>(ns)) {
          result = dyn_cast<NamespaceDecl>(ns)->lookup(decl->getDeclName());
        } else {
          result = dyn_cast<NamespaceAliasDecl>(ns)->getNamespace()->lookup(
              decl->getDeclName());
        }
      } break;
      case NestedNameSpecifier::Kind::Type:
        result = spec.getAsRecordDecl()->lookup(decl->getDeclName());
        break;
      case NestedNameSpecifier::Kind::Global:
        result = ast.getTranslationUnitDecl()->lookup(decl->getDeclName());
        break;
      case NestedNameSpecifier::Kind::Null:
      case NestedNameSpecifier::Kind::MicrosoftSuper: break;
    }
#endif

    if (result.empty()) {
      llvm::errs() << "[WARNING] Lookup failed\n";
    } else {
      // TODO: check that using the first decl is always correct
      TraverseDecl(result.front());
    }
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr* expr) {
    NamedDecl* found_decl = expr->getFoundDecl();
    NamedDecl* decl       = expr->getDecl();
    if (decl != found_decl) {
      if (auto* shadow_decl = dyn_cast<UsingShadowDecl>(expr->getFoundDecl())) {
        // We don't use Traverse* since this is an implicit declaration
        VisitUsingShadowDecl(shadow_decl);
      } else {
        TraverseDecl(expr->getFoundDecl());
      }
    }
    TraverseDecl(decl);
    if (expr->hasQualifier()) {
      VisitNestedNameSpecifier(expr->getQualifier());
    }
    return true;
  }

  bool VisitCallExpr(CallExpr* call) {
    Decl* callee = call->getCalleeDecl();
    if (auto fun = dyn_cast<FunctionDecl>(callee)) {
      if (fun->isTemplateInstantiation()) {
        TraverseDecl(fun->getPrimaryTemplate());
        return true;
      }
    }
    TraverseDecl(callee);
    return true;
  }
};

class ExtractFunctorDependenciesConsumer : public ASTConsumer {
 private:
  struct SourceInfo {
    std::unordered_set<const Decl*> used_decls;
    std::unordered_set<std::string> includes;
    CallExpr* parallel_for_call = nullptr;
    CXXRecordDecl* functor      = nullptr;
    LambdaExpr* lambda          = nullptr;
  };

  CompilerInstance& compiler;
  const std::vector<std::string> include_dirs;
  std::vector<const Decl*> toplevel_decls;
  std::vector<FunctionDecl*> parallel_for_decls;

  /**
   * @brief Retrieve the declarations used by the Functor type of a
   * krepe::parallel_for call, as well as other info
   *
   * @param translation_unit The translation unit declaration obtained after
   *        parsing the whole AST
   * @param fun The parallel_for instantiation we want to get infos on
   * @return The SourceInfo for this parallel_for instantiation
   */
  SourceInfo get_source_info(TranslationUnitDecl* translation_unit,
                             const FunctionDecl* fun) {
    const auto params = fun->parameters();
    if (params.size() != 3) {
      llvm::errs() << "krepe::parallel_for should have 3 parameter, got "
                   << params.size() << '\n';
      abort();
    }

    ParmVarDecl* functor          = params[2];
    CXXRecordDecl* functor_struct = get_struct_decl(functor->getType());
    if (!functor_struct) {
      llvm::errs() << "The third argument to krepe::parallel_for should be a "
                      "record type\n";
      abort();
    }

    FindUsedDeclsVisitor visitor(compiler, include_dirs);
    LambdaExpr* lambda = nullptr;
    CallExpr* fun_call = nullptr;
    if (functor_struct->isLambda()) {
      // TODO: see if traversing the tree for each parallel_for specialization
      // introduces a performance penalty compared to traversing once and
      // storing every lambda/parallel_for call
      FindLambdaExprVisitor lambda_visitor(functor_struct, fun);
      lambda_visitor.TraverseTranslationUnitDecl(translation_unit);
      std::tie(lambda, fun_call) = lambda_visitor.get_results();
      if (!lambda) {
        llvm::errs()
            << "Failed to find the LambdaExpr corresponding to a RecordDecl\n";
        abort();
      }
      if (!fun_call) {
        llvm::errs() << "Failed to find the CallExpr corresponding to a "
                        "krepe::parallel_for call\n";
        abort();
      }

      visitor.set_excluded_function(Decl::castFromDeclContext(
          lambda->getLambdaClass()->getDeclContext()));

      visitor.TraverseLambdaExpr(lambda);

    } else {
      visitor.TraverseDecl(functor_struct);
      for (CXXMethodDecl* method : functor_struct->methods()) {
        DeclarationName decl_name = method->getDeclName();
        if (decl_name.getNameKind() ==
                DeclarationName::NameKind::CXXOperatorName &&
            decl_name.getCXXOverloadedOperator() ==
                OverloadedOperatorKind::OO_Call) {
          visitor.TraverseDecl(method);
        }
      }
    }

    return {.used_decls        = visitor.get_used_decls(),
            .includes          = visitor.get_includes(),
            .parallel_for_call = fun_call,
            .functor           = functor_struct,
            .lambda            = lambda};
  }

  /**
   * @brief Determines whether a Declaration is extern, that is lives in a
   * header that should be #included
   */
  bool decl_is_extern(const Decl* decl) const {
    if (decl->isInStdNamespace()) {
      return true;
    }

    SourceLocation includeLoc = decl->getLocation();
    PresumedLoc prev;
    const SourceManager& source_manager = compiler.getSourceManager();

    while (includeLoc.isValid() && !source_manager.isInMainFile(includeLoc)) {
      PresumedLoc PLoc = source_manager.getPresumedLoc(includeLoc);
      FileID parentFid = source_manager.getFileID(includeLoc);
      includeLoc       = source_manager.getIncludeLoc(parentFid);
      prev             = PLoc;
    }

    if (prev.isInvalid()) {
      return false;
    }

    std::string_view filename = prev.getFilename();

    for (const auto& path : include_dirs) {
      if (filename.starts_with(path)) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Prints a sourceRange into a stream
   *
   * This function adds the trailing semicolon of records, which is not included
   * when calling `getSourceRange()`.
   *
   * @param stream The raw_ostream to print to
   * @param range The SourceRange of the Decl to print
   */
  void print_decl(llvm::raw_ostream& stream, SourceRange range) {
    const ASTContext& ast               = compiler.getASTContext();
    const SourceManager& source_manager = ast.getSourceManager();

    SourceLocation end = Lexer::findLocationAfterToken(
        range.getEnd(), tok::semi, source_manager, ast.getLangOpts(), true);
    if (end.isValid()) {
      range.setEnd(end.getLocWithOffset(-1));
    }

    if (range.isInvalid()) {
      return;
    }

    stream << Lexer::getSourceText(CharSourceRange::getTokenRange(range),
                                   source_manager, ast.getLangOpts());
  }

  static std::string_view access_specifier_name(AccessSpecifier spec) {
    switch (spec) {
      case AS_public: return "public";
      case AS_protected: return "protected";
      case AS_private: return "private";
      case AS_none: return "";
    }
  }

  /**
   * @brief Prints a Record to a stream
   *
   * Member variables, typedefs and constructors are always printed. Static
   * variables, static functions, and methods are only printed if they are used.
   *
   * @param stream The raw_ostream to print to
   * @param record The Record to print
   * @param used_decls an unordered_set containing the Decls used by the
   *        functor
   */
  void print_record(llvm::raw_ostream& stream, const CXXRecordDecl* record,
                    const std::unordered_set<const Decl*>& used_decls) {
    const PrintingPolicy& print_policy =
        compiler.getASTContext().getPrintingPolicy();

    if (record->isStruct()) {
      stream << "struct ";
    } else {
      stream << "class ";
    }
    stream << record->getName();

    // base specifiers
    if (record->getNumBases() > 0) {
      stream << " : ";
      bool first = true;
      for (const CXXBaseSpecifier base_spec : record->bases()) {
        if (!first) {
          stream << ", ";
        }
        first                       = false;
        AccessSpecifier access_spec = base_spec.getAccessSpecifierAsWritten();
        if (access_spec != AccessSpecifier::AS_none) {
          stream << access_specifier_name(access_spec) << ' ';
        }
        stream << base_spec.getType()->getAsRecordDecl()->getName();
      }
    }

    stream << " {\n";

    for (Decl* member : record->decls()) {
      if (member->isImplicit()) {
        continue;
      }

      if (const AccessSpecDecl* spec = dyn_cast<AccessSpecDecl>(member)) {
        stream << access_specifier_name(spec->getAccess()) << ":\n";
      } else if (isa<CXXConstructorDecl>(member)) {
        member->print(stream, print_policy);
      } else if (isa<VarDecl, FunctionDecl, RedeclarableTemplateDecl>(member)) {
        if (used_decls.contains(member)) {
          member->print(stream, print_policy);
          if (isa<VarDecl>(member)) {
            stream << ';';
          }
          stream << '\n';
        }
      } else {
        member->print(stream, print_policy);
        stream << ";\n";
      }
    }

    stream << "};\n\n";
  }

  /**
   * @brief Prints a Decl to a stream if it was used by a functor
   *
   * @param stream The raw_ostream to print to
   * @param decl The Decl to print
   * @param used_decls an unordered_set containing the Decls used by the
   *        functor
   */
  void process_decl(llvm::raw_ostream& stream, const Decl* decl,
                    std::unordered_set<const Decl*>& used_decls) {
    if (used_decls.erase(decl) == 0) {
      return;
    }

    if (const auto* ns = dyn_cast<NamespaceDecl>(decl)) {
      if (ns->isAnonymousNamespace()) {
        stream << "namespace {\n";
      } else if (ns->isInlineNamespace()) {
        stream << "inline namespace " << ns->getName() << " {\n";
      } else {
        stream << "namespace " << ns->getName() << " {\n";
      }

      const DeclContext* context = Decl::castToDeclContext(ns);
      for (const Decl* inner_decl : context->decls()) {
        process_decl(stream, inner_decl, used_decls);
      }

      stream << "}\n";
      return;
    }

    if (const auto* record = dyn_cast<CXXRecordDecl>(decl)) {
      print_record(stream, record, used_decls);
    } else {
      print_decl(stream, decl->getSourceRange());
      stream << '\n';
    }
  }

  /**
   * @brief Returns whether a struct has a callable default constructor
   *
   * This is necessary since `hasDefaultConstructor()` returns true when the
   * default constructor is explicitly deleted
   */
  static bool struct_has_default_ctor(const CXXRecordDecl* decl) {
    if (!decl->hasDefaultConstructor()) {
      return false;
    }

    for (const CXXConstructorDecl* ctor : decl->ctors()) {
      if (ctor->isDefaultConstructor() && !ctor->isDeleted()) {
        return true;
      }
    }

    return false;
  }

  /**
   * @brief Writes the declaration of an instance of a struct functor into a
   * stream
   *
   * For a struct of the following form
   * ```
   * struct Functor {
   *   int x;
   *   std::string s;
   *
   *   Functor(int, std::string) {}
   * };
   * ```
   *
   * This will write `auto functor = Functor();`
   *
   * If the default constructor is deleted, another suitable constructor will
   * be searched, in this case:
   *
   * `auto functor = Functor(int{}, std::string{});`
   */
  void write_struct_functor(llvm::raw_ostream& stream,
                            const CXXRecordDecl* functor,
                            const PrintingPolicy& print_policy) {
    std::string functor_initializer;
    llvm::raw_string_ostream initializer_stream(functor_initializer);
    if (!struct_has_default_ctor(functor)) {
      for (const CXXConstructorDecl* ctor : functor->ctors()) {
        if (ctor->isDeleted() || ctor->isCopyOrMoveConstructor()) {
          continue;
        }
        bool first_decl = true;
        for (const ParmVarDecl* param : ctor->parameters()) {
          QualType type = param->getType();
          // FIXME: this could be recursive A(B(int{}, C(float{}),
          // std::string{})
          if (type->isRecordType() &&
              !struct_has_default_ctor(type->getAsCXXRecordDecl())) {
            functor_initializer.clear();
            break;
          }
          if (!first_decl) {
            initializer_stream << ", ";
          }
          get_canonical_type(type).print(initializer_stream, print_policy);
          initializer_stream << "{}";
          first_decl = false;
        }
        if (!functor_initializer.empty()) {
          break;
        }
      }

      if (functor_initializer.empty()) {
        DiagnosticsEngine& diagnostics  = compiler.getDiagnostics();
        static unsigned no_ctor_diag_id = diagnostics.getCustomDiagID(
            DiagnosticsEngine::Warning,
            "[krepe auto extract plugin] Couldn't find a suitable "
            "constructor to build a replayed functor using defaulted "
            "values");
        static unsigned no_ctor_note_diag_id = diagnostics.getCustomDiagID(
            DiagnosticsEngine::Note, "For the following functor type: %0");

        diagnostics.Report(SourceLocation{}, no_ctor_diag_id);

        std::string functor_name;
        llvm::raw_string_ostream functor_name_stream(functor_name);
        functor->getNameForDiagnostic(functor_name_stream, print_policy, true);

        diagnostics.Report(functor->getBeginLoc(), no_ctor_note_diag_id)
            << functor_name;
      }
    }

    stream << "  ";
    functor->printQualifiedName(stream, print_policy);
    stream << " functor";
    if (!functor_initializer.empty()) {
      stream << '(' << functor_initializer << ')';
    }
    stream << ";\n";
  }

  /**
   * @brief Writes a main function replicating a given krepe::parallel_for
   * call
   *
   * This function will write the kokkos and krepe scope guards, a variable
   * for the policy, a variable for a defaulted functor and a parallel_for
   * call followed by a fence.
   *
   * The policy declaration has the form `auto policy = <policy_type>{@};`,
   * where "{@}" is a placeholder allowing the tool to write the correct
   * runtime arguments for the policy. When <policy_type> is a builtin type,
   * it is ommited to allow for declarations of the form `auto policy = 5;`.
   */
  void write_main_function(llvm::raw_ostream& stream,
                           const FunctionDecl* parallel_for_inst,
                           const CallExpr* parallel_for_call,
                           const CXXRecordDecl* functor_struct,
                           const LambdaExpr* lambda) {
    const PrintingPolicy& print_policy =
        compiler.getASTContext().getPrintingPolicy();
    stream << R"(
int main(int argc, char* argv[]) {
  krepe::ScopeGuard krepe_guard(argc, argv);
  Kokkos::ScopeGuard kokkos_guard(argc, argv);

)";

    // We write the policy before the functor in case the policy is referenced
    // inside the lambda, e.g.
    //
    // parallel_for("", team_policy,
    // KOKKOS_LAMBDA(decltype(team_policy)::member_type team) {}
    StringRef policy_name = "policy";
    if (lambda) {
      const Expr* policy_arg = parallel_for_call->getArgs()[1]->IgnoreCasts();
      if (auto* expr = dyn_cast<DeclRefExpr>(policy_arg)) {
        policy_name = expr->getFoundDecl()->getName();
      }
    }

    QualType policy_type =
        get_canonical_type(parallel_for_inst->getParamDecl(1)->getType());

    stream << "  auto " << policy_name << " = ";
    if (!policy_type->isBuiltinType()) {
      policy_type.print(stream, print_policy);
    }
    stream << "{@};\n";

    if (lambda) {
      // FIXME: Make this work with non default-constructible types
      for (const LambdaCapture& capture : lambda->captures()) {
        if (capture.capturesThis()) {
          continue;
        }
        const ValueDecl* captured_var = capture.getCapturedVar();
        stream << "  ";
        captured_var->getType().print(stream, print_policy);
        stream << ' ' << captured_var->getName();
        if (captured_var->getType()->isBuiltinType()) {
          stream << "{}";
        }
        stream << ";\n";
      }
      stream << "  auto functor = ";
      print_decl(stream, lambda->getSourceRange());
      stream << ";\n";
    } else {
      write_struct_functor(stream, functor_struct, print_policy);
    }

    stream << "  krepe::parallel_for(\"\", " << policy_name << ", functor);\n";
    stream << "  Kokkos::fence();\n";
    stream << "}\n";
  }

  /**
   * @brief Injects a call to `krepe::impl::register_replay_source` into a
   * parallel_for's body
   *
   * @param fun The `krepe::parallel_for` instantiation to modify
   * @param replay_source A string containing the source to be used to replay
   * this parallel_for call
   */
  void patch_parallel_for(FunctionDecl* fun, const std::string& replay_source) {
    ASTContext& ast = compiler.getASTContext();

    const NamespaceDecl* impl_namespace =
        dyn_cast<NamespaceDecl>(get_enclosed_decl(
            fun->getEnclosingNamespaceContext(), "impl", ast,
            [](const Decl* decl) { return isa<NamespaceDecl>(*decl); }));
    if (!impl_namespace) {
      llvm::errs() << "Failed to find the krepe::impl namespace\n";
      abort();
    }

    FunctionDecl* register_source_fun = dyn_cast<FunctionDecl>(
        get_enclosed_decl(Decl::castToDeclContext(impl_namespace),
                          "register_replay_source", ast, [](const Decl* decl) {
                            const auto* fun = dyn_cast<FunctionDecl>(decl);
                            return fun && fun->getNumParams() == 1;
                          }));

    if (!register_source_fun) {
      llvm::errs() << "Failed to find the krepe::impl::register_replay_source "
                      "function\n";
      abort();
    }

    Stmt* body         = fun->getBody();
    SourceLocation loc = body->getBeginLoc();

    // Build the string literal containing the replayed source code
    uint64_t replay_source_size = replay_source.size();
    unsigned int nb_bits =
        sizeof(replay_source_size) - std::countl_zero(replay_source_size);
    llvm::APInt llvm_replay_source_size(nb_bits, replay_source_size);

    QualType source_literal_type = ast.getConstantArrayType(
        ast.getConstType(ast.CharTy), llvm_replay_source_size,
        IntegerLiteral::Create(ast, llvm_replay_source_size,
                               ast.UnsignedLongLongTy, loc),
        ArraySizeModifier::Normal, 0);
    StringLiteral* source_code =
        StringLiteral::Create(ast, replay_source, StringLiteralKind::Ordinary,
                              false, source_literal_type, loc);

    // We create another if statement instead of finding and modifying the
    // existing one to avoid relying too much on assumptions about the code's
    // structure. We only assume that the body of the function will be a
    // CompoundStmt with the first child being the DeclStmt of a VarDecl for a
    // boolean which is true when this parallel_for invocation will trigger a
    // dump
    CompoundStmt* body_stmt = dyn_cast<CompoundStmt>(body);
    DeclStmt* will_dump_decl_stmt =
        dyn_cast<DeclStmt>(*body_stmt->child_begin());
    VarDecl* will_dump_decl =
        dyn_cast<VarDecl>(will_dump_decl_stmt->getSingleDecl());
    DeclRefExpr* will_dump_expr =
        DeclRefExpr::Create(ast, NestedNameSpecifierLoc{}, SourceLocation{},
                            dyn_cast<ValueDecl>(will_dump_decl), false, loc,
                            will_dump_decl->getType(), VK_LValue);

    IfStmt* if_stmt = IfStmt::Create(
        ast, loc, IfStatementKind::Ordinary, nullptr, nullptr, will_dump_expr,
        loc, loc,
        build_function_call(compiler, register_source_fun, {source_code}));

    std::vector<Stmt*> statements = {will_dump_decl_stmt, if_stmt};
    statements.insert(statements.end(), ++body_stmt->child_begin(),
                      body_stmt->child_end());
    CompoundStmt* injected_stmt =
        CompoundStmt::Create(ast, statements, FPOptionsOverride{}, loc, loc);

    fun->setBody(injected_stmt);
  }

 public:
  ExtractFunctorDependenciesConsumer(
      CompilerInstance& Instance, const std::vector<std::string>& include_dirs)
      : compiler(Instance), include_dirs(include_dirs) {}

  bool HandleTopLevelDecl(DeclGroupRef group) override {
    for (DeclGroupRef::iterator it = group.begin(), end = group.end();
         it != end; ++it) {
      Decl* decl = *it;

      FunctionDecl* function = dyn_cast<FunctionDecl>(decl);
      if (function && function->getIdentifier() &&
          function->getName() == "parallel_for") {
        const NamespaceDecl* krepe_namespace =
            dyn_cast<NamespaceDecl>(function->getEnclosingNamespaceContext());
        if (krepe_namespace && !krepe_namespace->isAnonymousNamespace() &&
            krepe_namespace->getName() == "krepe") {
          parallel_for_decls.push_back(function);
          continue;
        }
      }

      if (!decl_is_extern(decl)) {
        toplevel_decls.push_back(decl);
      }
    }

    return true;
  }

  void HandleTranslationUnit(ASTContext& ast) override {
    TranslationUnitDecl* tu_decl = ast.getTranslationUnitDecl();

    for (FunctionDecl* decl : parallel_for_decls) {
      auto [decls, includes, call, functor, lambda] =
          get_source_info(tu_decl, decl);
      includes.insert("Kokkos_Core.hpp");
      includes.insert("krepe/replayer.hpp");

      std::string source;
      llvm::raw_string_ostream source_stream(source);

      for (const std::string& path : includes) {
        source_stream << "#include <" << path << ">\n";
      }

      for (const Decl* toplevel_decl : toplevel_decls) {
        process_decl(source_stream, toplevel_decl, decls);
      }

      write_main_function(source_stream, decl, call, functor, lambda);

      source += '\0';
      patch_parallel_for(decl, source);
    }
  }
};

class ExtractFunctorDependenciesAction : public PluginASTAction {
 private:
  std::vector<std::string> include_dirs;
  std::unordered_set<std::filesystem::path> local_include_dirs;

 protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& compiler,
                                                 llvm::StringRef) override {
    HeaderSearchOptions& header_opts =
        compiler.getInvocation().getHeaderSearchOpts();

    for (const auto& entry : header_opts.UserEntries) {
      if (!local_include_dirs.contains(entry.Path)) {
        include_dirs.push_back(entry.Path);
      }
    }
    return std::make_unique<ExtractFunctorDependenciesConsumer>(compiler,
                                                                include_dirs);
  }

  bool ParseArgs(const CompilerInstance&,
                 const std::vector<std::string>& args) override {
    using size_type = std::string::size_type;

    std::string_view prefix = "include-dirs=";

    for (const std::string& arg : args) {
      size_type pos = arg.find(prefix);
      if (pos != 0) {
        llvm::errs() << "Unexpected arg " << arg << '\n';
        abort();
      }
      std::string_view arg_view = arg;
      std::string_view value    = arg_view.substr(prefix.size());

      size_type next_sep = 0;
      do {
        value                 = value.substr(next_sep);
        next_sep              = value.find(':');
        std::string_view path = value.substr(0, next_sep);
        local_include_dirs.emplace(path);
      } while (next_sep != std::string::npos);
    }
    return true;
  }

  PluginASTAction::ActionType getActionType() override {
    return AddBeforeMainAction;
  }
};

}  // namespace

static FrontendPluginRegistry::Add<ExtractFunctorDependenciesAction> Y(
    "auto_extract",
    "Patch krepe::parallel_for calls to extract a self-contained reproducer "
    "call when dumping");
