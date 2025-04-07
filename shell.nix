        

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
    pkgs.python311
    pkgs.stdenv.cc.cc.lib
    pkgs.python311Packages.libsbml
    pkgs.python311Packages.sympy
    pkgs.python311Packages.scipy
    pkgs.python311Packages.numpy
    pkgs.python311Packages.networkx
    #pkgs.python311Packages.jax
    #pkgs.python311Packages.jaxlib
    #pkgs.python311Packages.equinox

    pkgs.python311Packages.torch
    #pkgs.python311Packages.jaxlibWithCuda
    #pkgs.cudaPackages.cuda_cccl.dev
    pkgs.python311Packages.tensorflow-probability


    pkgs.python311Packages.pip
    pkgs.python311Packages.requests




  ];
  shellHook = ''
    # Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
    # See https://pip.pypa.io/en/stable/user_guide/#environment-variables.
    export EXTRA_CCFLAGS="-I/usr/include"
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python311.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH

  '';
}

