import "regent"
local c = regentlib.c
terralib.includepath = terralib.includepath .. ";/usr/local/opt/openblas/include/" .. ";/usr/local/opt/lapack/include/"

terralib.linklibrary("/usr/local/opt/openblas/lib/libopenblas.dylib")
local blas = terralib.includec("cblas.h")

terralib.linklibrary("/usr/local/opt/openblas/lib/liblapack.dylib")
local lapack = terralib.includec("lapacke.h")


task main()

  var a: double[2]
  a[0] = 1
  a[1] = 2

  var b: double[2]
  b[0] = 1
  b[1] = 1

  var dot = blas.cblas_ddot(2, a, 1, b, 1)
  c.printf("%f\n", dot)
end


regentlib.start(main)
