"""
Classes for reading isochrones, with different ages packed into a single file.
Examples are the .iso.cmd files from the MIST models, or the files produced by
the PADOVA web interface.
"""

from numpy import array, where, int32, float64, zeros, append, empty, log10
import re

class MIST:

    """
    Reads MIST CMD files. Code is modified version of the class in
    https://github.com/jieunchoi/MIST_codes/blob/master/scripts/read_mist_models.py
    """

    def __init__(self, filename, verbose=True):
    
        """
        
        Args:
            filename: the name of .iso.cmd file.
        
        Usage:
            >> isocmd = readmodelgrids.MIST('MIST_v1.0_feh_p0.00_afe_p0.0_vvcrit0.4.iso.cmd')
            >> age_ind = isocmd.age_index(7.0)
            >> B = isocmd.isocmds[age_ind]['Bessell_B']
            >> V = isocmd.isocmds[age_ind]['Bessell_V']
            >> plt.plot(B-V, V) #plot the CMD for logage = 7.0
        
        Attributes:
            version         Dictionary containing the MIST and MESA version numbers.
            photo_sys       Photometric system. 
            abun            Dictionary containing Yinit, Zinit, [Fe/H], and [a/Fe] values.
            Av_extinction   Av for CCM89 extinction.
            rot             Rotation in units of surface v/v_crit.
            ages            List of ages.
            num_ages        Number of ages.
            hdr_list        List of column headers.
            isocmds         Data.
        
        """
        
        self.filename = filename
        if verbose:
            print('Reading in: ' + self.filename)
            
        self.version, self.photo_sys, self.abun, self.Av_extinction, self.rot, self.ages, self.num_ages, self.hdr_list, self.isocmds = self.read_isocmd_file()
    
    def read_isocmd_file(self):

        """

        Reads in the .iso.cmd file.
        
        Args:
            filename: the name of .iso.cmd file.
        
        """
        
        #open file and read it in
        with open(self.filename) as f:
            content = [line.split() for line in f]
        version = {'MIST': content[0][-1], 'MESA': content[1][-1]}
        photo_sys = ' '.join(content[2][4:])
        abun = {content[4][i]:float(content[5][i]) for i in range(1,5)}
        rot = float(content[5][-1])
        num_ages = int(content[7][-1])
        Av_extinction = float(content[8][-1])
        
        #read one block for each isochrone
        isocmd_set = []
        ages = []
        counter = 0
        data = content[10:]
        for i_age in range(num_ages):
            #grab info for each isochrone
            num_eeps = int(data[counter][-2])
            num_cols = int(data[counter][-1])
            hdr_list = data[counter+2][1:]
            formats = tuple([int32]+[float64 for i in range(num_cols-1)])
            isocmd = zeros((num_eeps),{'names':tuple(hdr_list),'formats':tuple(formats)})
            #read through EEPs for each isochrone
            for eep in range(num_eeps):
                isocmd_chunk = data[3+counter+eep]
                isocmd[eep]=tuple(isocmd_chunk)
            isocmd_set.append(isocmd)
            ages.append(isocmd[0][1])
            counter+= 3+num_eeps+2
        return version, photo_sys, abun, Av_extinction, rot, ages, num_ages, hdr_list, isocmd_set

    def age_index(self, age):
        
        """

        Returns the index for the user-specified age.
        
        Args:
            age: the base-10 logarithm of the age of the isochrone.
        
        """
        
        diff_arr = abs(array(self.ages) - age)
        age_index = where(diff_arr == min(diff_arr))[0][0]
        
        if ((age > max(self.ages)) | (age < min(self.ages))):
            print('The requested age is outside the range. Try between ' + str(min(self.ages)) + ' and ' + str(max(self.ages)))
            
        return age_index

class PADOVA:

    """
    Reads PADOVA CMD files (http://stev.oapd.inaf.it/cmd). The code is taken and modified from
    https://github.com/jieunchoi/MIST_codes/blob/master/scripts/read_mist_models.py
    """

    def __init__(self, filename, verbose=True, colibri=True):
    
        """
        
        Args:
            filename: the name of .iso.cmd file.

        Keywords:
            verbose: if True print out information while reading file
            colibri: if True isochrones table is assumed to be in PARSEC v1.2S + COLIBRI PR16 format,
            otherwise PARSEC version 1.2S is assumed.
        
        Usage:
            >> isocmd = readmodelgrids.PADOVA('PARSEC_1.2S_feh_p0.00_GGBPGRP.iso.cmd')
            >> age_ind = isocmd.age_index(7.0)
            >> G = isocmd.isocmds[age_ind]['G']
            >> G_BP = isocmd.isocmds[age_ind]['G_BP']
            >> G_RP = isocmd.isocmds[age_ind]['G_RP']
            >> plt.plot(G_BP-G_RP, G) #plot the CMD for logage = 7.0
        
        Attributes:
            version         String containing the version information
            photo_sys       Photometric system. 
            abun            Dictionary containing Yinit, Zinit, [Fe/H], and [a/Fe] values.
            Av_extinction   Av (Cardelli et al. (1989) + O'Donnell (1994) with RV=3.1.)
            ages            List of ages.
            num_ages        Number of ages.
            hdr_list        List of column headers.
            isocmds         Data.
        
        """
        
        self.filename = filename
        self.colibri = colibri
        if verbose:
            print('Reading in: ' + self.filename)
            
        self.version, self.photo_sys, self.abun, self.Av_extinction, self.ages, self.num_ages, self.hdr_list, self.isocmds = self.read_isocmd_file()
    
    def read_isocmd_file(self):

        """

        Reads in the .iso.cmd file.
        
        Args:
            filename: the name of .iso.cmd file.
        
        """

        Zsun = 0.0152
        
        #open file and read it in
        with open(self.filename) as f:
            content = [line.split() for line in f]

        if self.colibri:
            dataStart = 8 # First line with table header
            version = ' '.join(content[1][2:])
            photo_sys = ' '.join(content[3][4:])
            abun = {'Z':float(content[dataStart+1][0]), '[M/H]':log10(float(content[dataStart+1][0])/0.0152)}
            #num_ages = int(content[7][-1])
            Av_extinction = float(content[4][-1].split('=')[-1])
            comment = re.compile('^#')
            zstr = re.compile('Zini')
        else:
            version = ' '.join(content[1][2:])
            photo_sys = ' '.join(content[3][4:])
            Av_extinction = float(content[5][-1].split('=')[-1])
            if (Av_extinction>0):
                dataStart = 15
            else:
                dataStart = 14
            abun = {'Z':float(content[dataStart+1][0]), '[M/H]':log10(float(content[dataStart+1][0])/0.0152)}
            #num_ages = int(content[7][-1])
            comment = re.compile('^#')
            zstr = re.compile('Z')

        #read one block for each isochrone
        isocmd = None
        isocmd_set = []
        ages = []
        data = content[dataStart:]
        for item in data:
            if comment.match(item[0]):
                if zstr.match(item[1]):
                    if not (isocmd is None):
                        isocmd_set.append(isocmd)
                        if self.colibri:
                            ages.append(log10(isocmd[0][1]))
                        else:
                            ages.append(isocmd[0][1])
                    hdr_list = item[1:]
                    if self.colibri:
                        formats = tuple([float64 for i in range(7)] + [int32] + [float64 for i in
                            range(4)] + [int32] + [float64 for i in range(16)])
                    else:
                        formats = tuple([float64 for i in range(12)] + [int32])
                    isocmd = empty((0),{'names':tuple(hdr_list),'formats':tuple(formats)})
                continue
            if (float(item[4])>-9.999): 
                isocmd = append(isocmd, array(tuple(item), dtype=isocmd.dtype))
        if self.colibri:
            ages.append(log10(isocmd[0][1]))
        else:
            ages.append(isocmd[0][1])
        isocmd_set.append(isocmd)

        return version, photo_sys, abun, Av_extinction, ages, len(ages), hdr_list, isocmd_set

    def age_index(self, age):
        
        """

        Returns the index for the user-specified age.
        
        Args:
            age: the base-10 logarithm of the age of the isochrone.
        
        """
        
        diff_arr = abs(array(self.ages) - age)
        age_index = where(diff_arr == min(diff_arr))[0][0]
        
        if ((age > max(self.ages)) | (age < min(self.ages))):
            print('The requested age is outside the range. Try between ' + str(min(self.ages)) + ' and ' + str(max(self.ages)))
            
        return age_index
