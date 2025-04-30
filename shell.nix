        

let
  pkgs = import <nixpkgs> {
			config = {
				allowUnfree = true;
				#cudaSupport = true;
				};
			};
  diffrax = pkgs.callPackage ./diffrax.nix {pkgs=pkgs;};
  wadler-lindig = pkgs.callPackage ./wadler-lindig.nix {pkgs=pkgs;};
  sympy2jax= pkgs.callPackage ./sympy2jax.nix {pkgs=pkgs;};
in pkgs.mkShell {
  buildInputs = [
    pkgs.zlib
    pkgs.libsbml
    pkgs.python312
    pkgs.stdenv.cc.cc.lib
    pkgs.python312Packages.libsbml
    pkgs.python312Packages.sympy
    pkgs.python312Packages.scipy
    pkgs.python312Packages.numpy

    pkgs.python312Packages.networkx
    pkgs.python312Packages.jax
    pkgs.python312Packages.jaxlib
    #pkgs.python312Packages.jaxlibWithCuda
    pkgs.cudaPackages.cuda_cccl.dev
    pkgs.python312Packages.equinox
    pkgs.python312Packages.lineax
    pkgs.python312Packages.optimistix
    pkgs.python312Packages.torch
    pkgs.python312Packages.matplotlib



    pkgs.python312Packages.pip
    pkgs.python312Packages.requests
    wadler-lindig
    diffrax
    sympy2jax
   




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

