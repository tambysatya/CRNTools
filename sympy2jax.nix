/*
{
  lib,
  buildPythonPackage,
  fetchFromGitHub,

  # build-system
  hatchling,

  # dependencies
  equinox,
  jax,
  jaxtyping,
  typing-extensions,

  # tests
  beartype,
  pytest,
  python,
}:
*/
{pkgs}:
with pkgs; with pkgs.python312Packages;
buildPythonPackage rec {
  pname = "sympy2jax";
  version = "0.0.6";
  pyproject = true;

  src = pkgs.fetchFromGitHub {
    owner = "patrick-kidger";
    repo = "sympy2jax";
    tag = "v${version}";
    hash="sha256-9Y2zY3dKS+JOzKzYLhQ8JnHCTExq5a5BU/cEQxwdyWI=";
    #hash="sha256-F7OBN1lPXQX4VJXfdPT4r3Ley42r6j9gx2UoFELJDwk=";
  };

  build-system = with pkgs.python312Packages; [hatchling pythonRelaxDepsHook ];

  dependencies = [
    equinox
    jax
    jaxtyping
    typing-extensions
    typeguard
    lineax
  ];
  pythonImportsCheck = [ "lineax" ];
  pythonRemoveDeps = true;
  nativeBuildInputs = [pythonRelaxDepsHook];


  meta = {
    description = "Linear solvers in JAX and Equinox";
    homepage = "https://github.com/patrick-kidger/lineax";
    changelog = "https://github.com/patrick-kidger/lineax/releases/tag/v${version}";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ GaetanLepage ];
  };
}
