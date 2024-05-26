# imgtools-public
MATLAB tools, primarily for handling MRI data in a research setting - **not for clinical use**.

Copyright, 2019-2021, David Atkinson.

For PET-MR image reconstruction code from the EPSRC-funded Collaborative Computing Project, please go to https://www.ccpsynerbi.ac.uk

Licence
-------
The long-term intention is to make the code in this repository free for research and educational use, but not free for commercial use. I am in the slow process of trying to sort out the licence for this. So depite the name, the code is not public.

Warranty
--------

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

### Data GUI
`eshow` GUI to display and interact with 2D, 3D, RGB, temporal and complex data. Displays orthogonal slices. Good for inspecting values, control over windowing, creating montages or movies. (Note MATLAB now has `orthosliceViewer`, `sliceViewer` and Volume Viewer app which may be faster if you just wish to scroll through slices.) Now (on develop branch) allows ROI drawing on one figure and copying to another, maintaining position in patient space, provided `geom` structure has been passed in to `eshow`. 

`get_data_location` is a wrapper around either `uigetfile`, `uigetdir` or `dselect` and is intended to let the user manually select file(s) or folder once, then stores these as a preference variable. This avoids the need to keep re-entering files and paths. It also means that paths etc are not hard-coded within files. Useful when running test scripts.

### DICOM files and geometry representation
The live script file `explore_tform.mlx` demos some of the issues with coordinate systems, `tform`s etc.  Many of the  functions here have been developed over many years, primarily for use with DICOMs from Philips MR. Typically the reading functions need small alterations for new MR sequences, versions, scanners etc. The code would benefit from re-factoring, especially the file reading. Typical flow is:

    dinfo = datparse(dselect) ;
    vol = d2mat(dinfo,{'slice'},'op',fp') ;

`dmfrename` GUI to rename multiframe DICOMs (use `dselect` to avoid renaming.)

`dselect` GUI for selecting DICOM files from a folder. List can be constrained to DICOM type e.g. raw. Displays file information (files do not need to be renamed).

`datparse` Coverts files to `dinfo` structure. 

`d2mat` Takes a `dinfo` structure and reads into a MATLAB matrix with specified dimensions. Complicated to use and best understood from examples. If you are struggling, try `vol = d2mat(dinfo,{'frame'}) ;` for an unsorted output.

`vresample` reslice one data set into the geometry of another. Uses the `geom` structures which are created by `d2mat` when reading DICOM files - `geom` contains geometrical information such as `ImageOrentationPatient`. 

`dfastelemread` and `dfastinfo` Fast DICOM reading, useful for processing large numbers of files.

(Note MATLAB now has the app `dicomBrowser` and functions `dicomCollection` and `dicomreadVolume`).

`dicom2intrinsic` utility function linking LPH and MATLAB intrinsic coordinate systems.

### NIFTI
`n2mat` intended to extract image and DICOM-style geometry information from a NIFTI file. BEWARE that in tests, it proved necessary to reverse the slice order in one case. Also, the row-wise vs column-wise order in which the NIFTI file was created is likely to be important, but has to be specified manually.  

### ROIs
ROIs are an integral part of research in radiology but can be a pain to handle. The roadmap is to use the power of OsiriX/Horos to draw ROIs, and then import into MATLAB for analysis. To draw ROIs on MATLAB processed data, basic functionality is in `eshow` but it might be preferable to write the processed data as DICOM and read in to OsiriX/Horos. At the moment, `writeDicom.m` is still part of a private repo.


### Comments on code
A lot of the code requires at least the MATLAB Image Processing Toolbox. To check for other dependencies I use:

`[fList,pList] = matlab.codetools.requiredFilesAndProducts('eshow.m');`



