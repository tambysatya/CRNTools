        

let
  pkgs = import <nixpkgs> {
			config = {
				allowUnfree = true;
				#cudaSupport = true;
				};
			};
in pkgs.mkShell {
  buildInputs = [
    pkgs.libsbml
    pkgs.python312
    pkgs.stdenv.cc.cc.lib
    pkgs.python312Packages.libsbml
    pkgs.python312Packages.sympy
    pkgs.python312Packages.scipy
    pkgs.python312Packages.numpy
    pkgs.python312Packages.networkx
    #pkgs.python312Packages.jax
    #pkgs.python312Packages.jaxlib
    #pkgs.python312Packages.equinox

    pkgs.python312Packages.torch
    #pkgs.python312Packages.jaxlibWithCuda
    #pkgs.cudaPackages.cuda_cccl.dev


    pkgs.python312Packages.pip
    pkgs.python312Packages.requests




  ];
  shellHook = ''
    # Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
    # See https://pip.pypa.io/en/stable/user_guide/#environment-variables.
    export EXTRA_CCFLAGS="-I/usr/include"
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python312.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH

  '';
}

