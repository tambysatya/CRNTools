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
  pname = "diffrax";
  version = "0.7.0";
  pyproject = true;

  src = pkgs.fetchFromGitHub {
    owner = "patrick-kidger";
    repo = "diffrax";
    tag = "v${version}";
    #hash="sha256-F7OBN1lPXQX4VJXfdPT4r3Ley42r6j9gx2UoFELJDwk=";
    hash="sha256-TBXcwNFYU1C9Pa0KwJeqfVixnJNFdg77HRXqZlSEQjY=";
  };

  build-system = with pkgs.python312Packages; [hatchling pythonRelaxDepsHook ];

  dependencies = [
    equinox
    jax
    jaxtyping
    typing-extensions
    lineax
    optimistix
    typeguard
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
