#!env python
"""
Create an egg from python
"""
import os
import sys
import glob2
import shutil
import zipfile

shim_str_base = """
def __bootstrap__():
   global __bootstrap__, __loader__, __file__
   import sys, pkg_resources, imp
   __file__ = pkg_resources.resource_filename(__name__,'{0}')
   __loader__ = None; del __bootstrap__, __loader__
   imp.load_dynamic(__name__,__file__)
__bootstrap__()
"""

    


def run(SOURCE_EGG_INFO, BUILD_DIR, OUTPUT_EGG_FILE):
    # copy the egg info over originally
    try:
        shutil.rmtree(BUILD_DIR + "/EGG-INFO")
    except OSError:
        pass
    shutil.copytree(SOURCE_EGG_INFO, BUILD_DIR + "/EGG-INFO")

    native_str = ""

    ## Native file handling -- get the list of native objects
    for native_file in glob2.glob(os.path.join(BUILD_DIR,  "**/*.so")):
        so_name_and_path = native_file[len(BUILD_DIR)+1:]
        so_filename = os.path.basename(native_file)

        """
        Note this is insane, and I can't believe this is how eggs work. I have
        to write a manual import shim. 
        """


        shim_str =  shim_str_base.format(so_filename)
        shim_filename = os.path.join(os.path.dirname(native_file), 
                                    so_filename[:-2] + "py")
        f = open(shim_filename, 'w')
        f.write(shim_str)
        f.close()

        native_str += "%s\n" % so_name_and_path
    native_list_fid = open(os.path.join(BUILD_DIR, "EGG-INFO/native_libs.txt"), 'w')
    native_list_fid.write(native_str)
    native_list_fid.close()


    with zipfile.ZipFile(OUTPUT_EGG_FILE, 'w') as myegg:
        for filename in glob2.glob(os.path.join(BUILD_DIR, "**")):
            if not os.path.isdir(filename):
                archive_name = filename[len(BUILD_DIR)+1:]
                myegg.write(filename, archive_name)


if __name__ == "__main__":
    print "Building egg. Current working directory is", os.getcwd()
    
    # SOURCE_EGG_INFO="EGG-INFO"
    # BUILD_DIR="src/build/egg" 
    # OUTPUT_EGG_FILE="test.egg"
    print "THE SYS ARG IS", sys.argv
    source_egg_info = sys.argv[1]
    build_dir = sys.argv[2]
    output_egg_file = sys.argv[3]
    print "source_egg_info=", source_egg_info
    print "build_dir=", build_dir
    print "output_egg_file=", output_egg_file
    
    run(source_egg_info, build_dir, output_egg_file)
    
