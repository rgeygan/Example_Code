import attr

import itertools
import os
import numpy as np
import geopandas as gpd
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm

## NDA Prevents me from showing internals of code structure - this is an example of how to construct class structures
## with class methods

GLOBAL_VAR = 'foo'

##Global function
def counties(path: str):
    """
    Functions to generate list of counties. Currently, generates list of counties being used for Indigo regen model testing
    as of 8/12/2021. Can call this function as an argument inside load function
    """
    ##get dataframe

    ##make list from dataframe

    return counties

##Class to create data objects
@attr.s(frozen=False, slots=True, auto_attribs=True)
class ReferenceData:
    """
    Attributes describe ReferenceData object and contain counties over which data will be gathered.

    attribute data: this attribute should be instantiated as NONE when creating a ReferenceData object. Then use
    ReferenceData.data = ReferenceData.load() to load a dataframe and assign it to the data attribute.
    """
    year: list
    practice: str
    dataset_name: str
    counties: list
    crop: None
    data: None

    def load_data(self) -> pd.DataFrame:
        """
        Load USDA NASS ag census data at county level and parse into pandas dataframe
        Example values for practice arg: 'cover crops' and 'no till', which are converted to NASS labels with dict

                                        'COVER CROP PLANTED, (EXCL CRP)',
                                         'CONVENTIONAL TILLAGE',
                                         'CONSERVATION TILLAGE, NO-TILL',
                                         'CONSERVATION TILLAGE, (EXCL NO-TILL)'

        Example value for year arg: '2012', '2017'
        """

        print('Loading data...')

        ##Define local vars

        ##Find data and format

        return data

    @staticmethod
    def calc_tot_acres(args) -> list:
        """
        This function calculates total regen acres by model class and cdl class based on global variables results_path and
        class_column.
        :param county: list of counties to calculate total acreage
        :return: returns a list of lists with county, total regen acres, total related cdl acres, cdl crop class, regen class, and year
        """

        #Function to loop through tables and sum acreages

        return tmp

    ##Controller function that loads dataset given dataset_name and other params
    def load(self, **kwargs):

        option = kwargs.get('option',None)

        if self.dataset_name == 'foo':
            return self.load_nass()
        elif self.dataset_name == 'foo2':
            return self.load_indigo(option = option)
        elif self.dataset_name == 'foo3':
            return self.load_optis(option = option)

data_obj = ReferenceData(attributes)

##load data
data = data_obj.load()

##Class that holds functions - pass data objects and use class methods to return transformed data
@attr.s(frozen=False, slots=True, auto_attribs=True)
class TransformData:
    """
    Class methods here take ReferenceData objects as argument. Methods pull necessary information from
    ReferenceData class object attributes.

    attribute data: as with ReferenceData class, the data attribute here is instantiated empty as NONE type and then
    reassigned using adoption_rate and/or scale functions
    """
    data: None

    def transform(self,obj) -> pd.DataFrame:
        """
        This function takes in loaded data as a pd.Dataframe and converts from total acres of a regen practice to the adoption
        rate. Adoption rates can be calculated as "crop-specific" eg. regen acres divided by corn, soy, etc. acres, "available"
        """
        ##Hidden code - NDA prevents me from showing you the internals of this code

        return tmp

trans_obj = TransformData()

new_data = trans_obj.transform()
