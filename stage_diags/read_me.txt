A small collection of code that should be useful to get you started in analyzing some simulation output. If you run 

If you install the package with 


python setup.py develop


then you can edit the code in the folder without having to reinstall the package every time. You can also add new files for functions etc.

In a given python script, you can then import the functions for example as


from stage_diags import io_utils

params = io_utils.read_params("outdir_0001")
L = params["Lx"]


If you want to you can keep your own diagnostic functions in this package as well, so when you sync with git we can work on them together!

