README
======

This README relates to the imgtools files that are stored in a GitHub 
repository. Previously there was an intention to release files from here
as part of a zip package but this is being changed to a public Git repo. 

This private GitHub site is for collaborators and is: 
https://github.com/UCL/imgtools  

Some functions are being moved to the public repo imgtools-public:
https://github.com/UCL/imgtools-public

To check dependencies prior to moving, use for example,
[fList,pList] = matlab.codetools.requiredFilesAndProducts('eshow.m');

Licensing: This repository grew with no formal sharing or licensing arrangements other 
than an expectation that files might be used by research collaborators. The
longer term plan is that the repository will remain private and shared among 
collaborators, but, that certain folders will be released as part of a zip 
package under the Apache 2.0 license. 

Contributing: Generally contributors should expect to be using the 
Apache 2.0 license. Please develop on your own topic branch and not master.

External Packages: Some of the code here requires external files e.g. from the 
MATLAB FileExchange, NODDI, ISMRMRD etc. often with different licensing 
arrangements. Users should place this code in a separate location and add 
the locaton to their MATLAB path (see below).

A recommended folder layout is:
HOME/matlab/imgtools  - the contents of this repository or zip file
HOME/matlab/external  - locaton of user downloaded files
HOME/matlab/personal  - locaton of user-specifc files (web access tokens)


Part A outlines changes to the MATLAB path you need to make once.
Part B outlines the external files you may need.

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



A) MATLAB Path
--------------
Add the imgtools files and folders to your MATLAB path:

For ZIP file users:

  1) Unzip the contents of the zip file into a folder. As an example, suppose
     you unzipped into the folder C:\home\matlab\imgtools

  2) Add the folder to your MATLAB path. 
     Type:
      edit startup
     and add the line below (modifying the folder name if necessary):
      addpath(genpath('C:\home\matlab\imgtools'))

  3) Save the edited startup file, close and restart MATLAB. You only need to do this once.


  
For GIT repository users. Either follow the above, or see the instructions 
in the file set_imgtools_path.m


B) External Files
-----------------
Some functions in imgtools require third party code to be downloaded. The downloaded code will 
need to be on your MATLAB path. Below is a list of sources used. We are looking for 
ways to streamline downloading and dependency checks. To facilitate future distribution, downloaded 
code is being moved to a separate GitHub site (imgtools-external) so that this imgtools site 
contains code we have developed and thus has less chance of licensing/copyright issues.

B0-NICE: B0 unwrapping from MATLAB FileExchange MRM 2015

 PROPACK: The RDDR and RPCA code calls the PROPACK libraries (V 1.1) from:
   http://soi.stanford.edu/~rmunk/PROPACK/ 
 
  RPCA is based on code from http://perception.csl.illinois.edu/matrix-rank/sample_code.html
  PROPACK can also be downloaded as part of a zip file from this site.

 MIRT: RDDR uses the MIRT Medical Image Registration Toolbox for the residual
 complexity method:
   https://sites.google.com/site/myronenko/research/mirt

 MISST: Microstructure Imaging Sequence Simulation Toolbox, download from NITRC site
   http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.MISST


NODDI processing uses the NODDI code from Gary Zhang:
  http://cmic.cs.ucl.ac.uk/mig/index.php?n=Tutorial.NODDImatlab

EPG: Brian Hargreaves Extended Phase Graph code:
  http://web.stanford.edu/~bah/software/epg/

ISMRMRD: https://github.com/ismrmrd/ismrmrd

Note that the SpinCalc.m function which is included here, comes from the MATLAB FileExchange:
http://www.mathworks.co.uk/matlabcentral/fileexchange/41562-spinconv/content/SpinCalc.m

------------------

Exporting a zip file from the Git repository.
Switch to the master branch and inspect .gitattributes to make sure the files
you wish to exclude are excluded.

Either use git bash and then (for tag 1.2):  
git archive -o imgtoolsv1p2.zip 1.2
 or, 
TortoiseGit -> Show Log then Export from Context menu.


If you see a date here >> Thu Aug 31 17:17:24 2023 +0100  << , it was automatically inserted when the README.txt file 
was exported from the Git repository. If you do not see a date, you are probably looking
at the file as part of the repository.


