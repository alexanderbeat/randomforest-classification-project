
## Final Project Submission

Please fill out:
* Student name: Alex Beat
* Student pace: part time
* Scheduled project review date/time: 05/10/20 @10am pacific
* Instructor name: James Irving
* Blog post URL: NA


# TABLE OF CONTENTS 

*Click to jump to matching Markdown Header.*<br><br>

<font size=4rem>
    
- [Introduction](#INTRODUCTION)<br>
- **[OBTAIN](#OBTAIN)**<br>
- **[SCRUB](#SCRUB)**<br>
- **[EXPLORE](#EXPLORE)**<br>
- **[MODEL](#MODEL)**<br>
- **[iNTERPRET](#iNTERPRET)**<br>
- [Conclusions/Recommendations](#CONCLUSIONS-&-RECOMMENDATIONS)<br>
</font>
___


## PROCESS CHECKLIST


1. **[OBTAIN](#OBTAIN)**
    - Import data, inspect, check for datatypes to convert and null values
    - Display header and info.
    - Drop any unneeded columns, if known (`df.drop(['col1','col2'],axis=1,inplace=True`)
    <br><br>

2. **[SCRUB](#SCRUB)**
    - Recast data types, identify outliers, check for multicollinearity, normalize data**
    - Check and cast data types
        - [ ] Check for #'s that are store as objects (`df.info()`,`df.describe()`)
            - when converting to #'s, look for odd values (like many 0's), or strings that can't be converted.
            - Decide how to deal weird/null values (`df.unique()`, `df.isna().sum()`)
            - `df.fillna(subset=['col_with_nulls'],'fill_value')`, `df.replace()`
        - [ ] Check for categorical variables stored as integers.
            - May be easier to tell when you make a scatter plotm or `pd.plotting.scatter_matrix()`
            
    - [ ] Check for missing values  (df.isna().sum())
        - Can drop rows or colums
        - For missing numeric data with median or bin/convert to categorical
        - For missing categorical data: make NaN own category OR replace with most common category
    - [ ] Check for multicollinearity
        - Use seaborn to make correlation matrix plot 
        - Good rule of thumb is anything over 0.75 corr is high, remove the variable that has the most correl with the largest # of variables
    - [ ] Normalize data (may want to do after some exploring)
        - Most popular is Z-scoring (but won't fix skew) 
        - Can log-transform to fix skewed data
    
3. **[EXPLORE](#EXPLORE)**
    - [ ] Check distributions, outliers, etc**
    - [ ] Check scales, ranges (df.describe())
    - [ ] Check histograms to get an idea of distributions (df.hist()) and data transformations to perform.
        - Can also do kernel density estimates
    - [ ] Use scatter plots to check for linearity and possible categorical variables (`df.plot("x","y")`)
        - categoricals will look like vertical lines
    - [ ] Use `pd.plotting.scatter_matrix(df)` to visualize possible relationships
    - [ ] Check for linearity.
   
4. **[MODEL](#MODEL)**

    - **Fit an initial model:** 
        - Run an initial model and get results

    - **Holdout validation / Train/test split**
        - use sklearn `train_test_split`
    
5. **[iNTERPRET](#iNTERPRET)**
    - **Assessing the model:**
        - Assess parameters (slope,intercept)
        - Check if the model explains the variation in the data (RMSE, F, R_square)
        - *Are the coeffs, slopes, intercepts in appropriate units?*
        - *Whats the impact of collinearity? Can we ignore?*
        <br><br>
    - **Revise the fitted model**
        - Multicollinearity is big issue for lin regression and cannot fully remove it
        - Use the predictive ability of model to test it (like R2 and RMSE)
        - Check for missed non-linearity
        
- **Interpret final model and draw >=3 conclusions and recommendations from dataset**

# INTRODUCTION

> Problem: 
Predict which Tanzanian water pumps are functional, which need repairs, and which don't work at all. Predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.

> Audience: Main audience is the pump management companies who handles repairs and construction, along with city officials who work with companies to make decisions on pump location and technology for the pumps used. 

> Business Questions: Do older pumps need more repairs or are more prone to breaking? What is the relationship between population and working or nonworking pumps? Which type of pump to avoid?



```python
# # > Business outlook/questions:
# # "How did you pick the question(s) that you did?"
# based on which graphs and features were most important for determining class

# # "Why are these questions important from a business perspective?"
# they will help us to decide which locations, population concentrations, and types of pumps to focus on

# # "How did you decide on the data cleaning options you performed?"
# mainly, based on exploring the data to find a lot of repeat categoricals taht became redundant 
# and useless information that was either not effective in determining class, such as constant variables 
# or null values. 

# # "Why did you choose a given method or library?"
# i used the sklearn library of classification algorithms such as random forest for their resilience to
# overfitting and xgbrf for the same reason. 

# # "Why did you select those visualizations and what did you learn from each of them?"
# i used the null value visualization which helped me to see that some columsn were matching in 
# missing values from same rows. partial dependence plots were useful in determining how the most important 
# features were effecting each classification to better understand why. 

# # "Why did you pick those features as predictors?"
# those were shown to be the most important and impactful in determining the class of the pumps, especially
# once followed up with partial dependence plots for further explainability. 

# # "How would you interpret the results?"
# see interpretation at end of notebook

# # "How confident are you in the predictive quality of the results?"
# mostly confident about results for classification of working and nonworking pumps. functional but needing repairs 
# classification needs to be tuned more. 

# # "What are some of the things that could cause the results to be wrong?"
# there were more weird values in the data, so, dirty data. needing more feature engineering in the future. 
```

## feature desc.


```python
# """
# id number ** - don't need, will match with index
# amount_tsh - Total static head (amount water available to waterpoint)
# date_recorded - The date the row was entered **
# funder - Who funded the well **
# gps_height - Altitude of the well
# installer - Organization that installed the well **
# longitude - GPS coordinate
# latitude - GPS coordinate 
# wpt_name - Name of the waterpoint if there is one **
# num_private - **
# basin - Geographic water basin **
# subvillage - Geographic location **
# region - Geographic location **
# region_code - Geographic location (coded) ++
# district_code - Geographic location (coded) ++
# lga - Geographic location **
# ward - Geographic location **
# population - Population around the well
# public_meeting - True/False **
# recorded_by - Group entering this row of data **
# scheme_management - Who operates the waterpoint ++ **
# scheme_name - Who operates the waterpoint ++ **
# permit - If the waterpoint is permitted **
# construction_year - Year the waterpoint was constructed
# extraction_type - The kind of extraction the waterpoint uses ++ **
# extraction_type_group - The kind of extraction the waterpoint uses ++ **
# extraction_type_class - The kind of extraction the waterpoint uses ++
# management - How the waterpoint is managed ++ **
# management_group - How the waterpoint is managed ++
# payment - What the water costs **
# payment_type - What the water costs ++
# water_quality - The quality of the water **
# quality_group - The quality of the water ++
# quantity - The quantity of water **
# quantity_group - The quantity of water ++
# source - The source of the water **
# source_type - The source of the water ++
# source_class - The source of the water ++ **
# waterpoint_type - The kind of waterpoint **
# waterpoint_type_group - The kind of waterpoint"""
```

# OBTAIN DATA


```python
# Import data, inspect, check for datatypes to convert and null values
# Display header and info.
# Drop any unneeded columns, if known (df.drop(['col1','col2'],axis=1,inplace=True)
```


```python
from fsds_100719.imports import *
```

    fsds_1007219  v0.7.16 loaded.  Read the docs: https://fsds.readthedocs.io/en/latest/ 



<style  type="text/css" >
</style><table id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26" ><caption>Loaded Packages and Handles</caption><thead>    <tr>        <th class="col_heading level0 col0" >Handle</th>        <th class="col_heading level0 col1" >Package</th>        <th class="col_heading level0 col2" >Description</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row0_col0" class="data row0 col0" >dp</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row0_col1" class="data row0 col1" >IPython.display</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row0_col2" class="data row0 col2" >Display modules with helpful display and clearing commands.</td>
            </tr>
            <tr>
                                <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row1_col0" class="data row1 col0" >fs</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row1_col1" class="data row1 col1" >fsds_100719</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row1_col2" class="data row1 col2" >Custom data science bootcamp student package</td>
            </tr>
            <tr>
                                <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row2_col0" class="data row2 col0" >mpl</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row2_col1" class="data row2 col1" >matplotlib</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row2_col2" class="data row2 col2" >Matplotlib's base OOP module with formatting artists</td>
            </tr>
            <tr>
                                <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row3_col0" class="data row3 col0" >plt</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row3_col1" class="data row3 col1" >matplotlib.pyplot</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row3_col2" class="data row3 col2" >Matplotlib's matlab-like plotting module</td>
            </tr>
            <tr>
                                <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row4_col0" class="data row4 col0" >np</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row4_col1" class="data row4 col1" >numpy</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row4_col2" class="data row4 col2" >scientific computing with Python</td>
            </tr>
            <tr>
                                <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row5_col0" class="data row5 col0" >pd</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row5_col1" class="data row5 col1" >pandas</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row5_col2" class="data row5 col2" >High performance data structures and tools</td>
            </tr>
            <tr>
                                <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row6_col0" class="data row6 col0" >sns</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row6_col1" class="data row6 col1" >seaborn</td>
                        <td id="T_5914f6e4_930f_11ea_9fb4_0026bb4edb26row6_col2" class="data row6 col2" >High-level data visualization library based on matplotlib</td>
            </tr>
    </tbody></table>



        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        


    [i] Pandas .iplot() method activated.


## Set plot default and DF show all cols


```python
plt.style.use('seaborn-notebook')
pd.set_option('display.max_columns',0)
```

## Load data


```python
df = pd.read_csv('trainingsetvalues.csv')
training_labels = pd.read_csv('trainingsetlabels.csv')
testdata = pd.read_csv('testsetvalues.csv')
```


```python
df.columns
```




    Index(['id', 'amount_tsh', 'date_recorded', 'funder', 'gps_height',
           'installer', 'longitude', 'latitude', 'wpt_name', 'num_private',
           'basin', 'subvillage', 'region', 'region_code', 'district_code', 'lga',
           'ward', 'population', 'public_meeting', 'recorded_by',
           'scheme_management', 'scheme_name', 'permit', 'construction_year',
           'extraction_type', 'extraction_type_group', 'extraction_type_class',
           'management', 'management_group', 'payment', 'payment_type',
           'water_quality', 'quality_group', 'quantity', 'quantity_group',
           'source', 'source_type', 'source_class', 'waterpoint_type',
           'waterpoint_type_group'],
          dtype='object')



## View data


```python
display(df)
display(training_labels)
display(testdata)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>amount_tsh</th>
      <th>date_recorded</th>
      <th>funder</th>
      <th>gps_height</th>
      <th>installer</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>wpt_name</th>
      <th>num_private</th>
      <th>basin</th>
      <th>subvillage</th>
      <th>region</th>
      <th>region_code</th>
      <th>district_code</th>
      <th>lga</th>
      <th>ward</th>
      <th>population</th>
      <th>public_meeting</th>
      <th>recorded_by</th>
      <th>scheme_management</th>
      <th>scheme_name</th>
      <th>permit</th>
      <th>construction_year</th>
      <th>extraction_type</th>
      <th>extraction_type_group</th>
      <th>extraction_type_class</th>
      <th>management</th>
      <th>management_group</th>
      <th>payment</th>
      <th>payment_type</th>
      <th>water_quality</th>
      <th>quality_group</th>
      <th>quantity</th>
      <th>quantity_group</th>
      <th>source</th>
      <th>source_type</th>
      <th>source_class</th>
      <th>waterpoint_type</th>
      <th>waterpoint_type_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>69572</td>
      <td>6000.0</td>
      <td>2011-03-14</td>
      <td>Roman</td>
      <td>1390</td>
      <td>Roman</td>
      <td>34.938093</td>
      <td>-9.856322</td>
      <td>none</td>
      <td>0</td>
      <td>Lake Nyasa</td>
      <td>Mnyusi B</td>
      <td>Iringa</td>
      <td>11</td>
      <td>5</td>
      <td>Ludewa</td>
      <td>Mundindi</td>
      <td>109</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>Roman</td>
      <td>False</td>
      <td>1999</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>pay annually</td>
      <td>annually</td>
      <td>soft</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>spring</td>
      <td>spring</td>
      <td>groundwater</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <td>1</td>
      <td>8776</td>
      <td>0.0</td>
      <td>2013-03-06</td>
      <td>Grumeti</td>
      <td>1399</td>
      <td>GRUMETI</td>
      <td>34.698766</td>
      <td>-2.147466</td>
      <td>Zahanati</td>
      <td>0</td>
      <td>Lake Victoria</td>
      <td>Nyamara</td>
      <td>Mara</td>
      <td>20</td>
      <td>2</td>
      <td>Serengeti</td>
      <td>Natta</td>
      <td>280</td>
      <td>NaN</td>
      <td>GeoData Consultants Ltd</td>
      <td>Other</td>
      <td>NaN</td>
      <td>True</td>
      <td>2010</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>wug</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>insufficient</td>
      <td>insufficient</td>
      <td>rainwater harvesting</td>
      <td>rainwater harvesting</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <td>2</td>
      <td>34310</td>
      <td>25.0</td>
      <td>2013-02-25</td>
      <td>Lottery Club</td>
      <td>686</td>
      <td>World vision</td>
      <td>37.460664</td>
      <td>-3.821329</td>
      <td>Kwa Mahundi</td>
      <td>0</td>
      <td>Pangani</td>
      <td>Majengo</td>
      <td>Manyara</td>
      <td>21</td>
      <td>4</td>
      <td>Simanjiro</td>
      <td>Ngorika</td>
      <td>250</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>Nyumba ya mungu pipe scheme</td>
      <td>True</td>
      <td>2009</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>pay per bucket</td>
      <td>per bucket</td>
      <td>soft</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>dam</td>
      <td>dam</td>
      <td>surface</td>
      <td>communal standpipe multiple</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <td>3</td>
      <td>67743</td>
      <td>0.0</td>
      <td>2013-01-28</td>
      <td>Unicef</td>
      <td>263</td>
      <td>UNICEF</td>
      <td>38.486161</td>
      <td>-11.155298</td>
      <td>Zahanati Ya Nanyumbu</td>
      <td>0</td>
      <td>Ruvuma / Southern Coast</td>
      <td>Mahakamani</td>
      <td>Mtwara</td>
      <td>90</td>
      <td>63</td>
      <td>Nanyumbu</td>
      <td>Nanyumbu</td>
      <td>58</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>NaN</td>
      <td>True</td>
      <td>1986</td>
      <td>submersible</td>
      <td>submersible</td>
      <td>submersible</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>dry</td>
      <td>dry</td>
      <td>machine dbh</td>
      <td>borehole</td>
      <td>groundwater</td>
      <td>communal standpipe multiple</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <td>4</td>
      <td>19728</td>
      <td>0.0</td>
      <td>2011-07-13</td>
      <td>Action In A</td>
      <td>0</td>
      <td>Artisan</td>
      <td>31.130847</td>
      <td>-1.825359</td>
      <td>Shuleni</td>
      <td>0</td>
      <td>Lake Victoria</td>
      <td>Kyanyamisa</td>
      <td>Kagera</td>
      <td>18</td>
      <td>1</td>
      <td>Karagwe</td>
      <td>Nyakasimbi</td>
      <td>0</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>0</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>other</td>
      <td>other</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>seasonal</td>
      <td>seasonal</td>
      <td>rainwater harvesting</td>
      <td>rainwater harvesting</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>59395</td>
      <td>60739</td>
      <td>10.0</td>
      <td>2013-05-03</td>
      <td>Germany Republi</td>
      <td>1210</td>
      <td>CES</td>
      <td>37.169807</td>
      <td>-3.253847</td>
      <td>Area Three Namba 27</td>
      <td>0</td>
      <td>Pangani</td>
      <td>Kiduruni</td>
      <td>Kilimanjaro</td>
      <td>3</td>
      <td>5</td>
      <td>Hai</td>
      <td>Masama Magharibi</td>
      <td>125</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>Water Board</td>
      <td>Losaa Kia water supply</td>
      <td>True</td>
      <td>1999</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>water board</td>
      <td>user-group</td>
      <td>pay per bucket</td>
      <td>per bucket</td>
      <td>soft</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>spring</td>
      <td>spring</td>
      <td>groundwater</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <td>59396</td>
      <td>27263</td>
      <td>4700.0</td>
      <td>2011-05-07</td>
      <td>Cefa-njombe</td>
      <td>1212</td>
      <td>Cefa</td>
      <td>35.249991</td>
      <td>-9.070629</td>
      <td>Kwa Yahona Kuvala</td>
      <td>0</td>
      <td>Rufiji</td>
      <td>Igumbilo</td>
      <td>Iringa</td>
      <td>11</td>
      <td>4</td>
      <td>Njombe</td>
      <td>Ikondo</td>
      <td>56</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>Ikondo electrical water sch</td>
      <td>True</td>
      <td>1996</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>pay annually</td>
      <td>annually</td>
      <td>soft</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>river</td>
      <td>river/lake</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <td>59397</td>
      <td>37057</td>
      <td>0.0</td>
      <td>2011-04-11</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>34.017087</td>
      <td>-8.750434</td>
      <td>Mashine</td>
      <td>0</td>
      <td>Rufiji</td>
      <td>Madungulu</td>
      <td>Mbeya</td>
      <td>12</td>
      <td>7</td>
      <td>Mbarali</td>
      <td>Chimala</td>
      <td>0</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>NaN</td>
      <td>False</td>
      <td>0</td>
      <td>swn 80</td>
      <td>swn 80</td>
      <td>handpump</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>pay monthly</td>
      <td>monthly</td>
      <td>fluoride</td>
      <td>fluoride</td>
      <td>enough</td>
      <td>enough</td>
      <td>machine dbh</td>
      <td>borehole</td>
      <td>groundwater</td>
      <td>hand pump</td>
      <td>hand pump</td>
    </tr>
    <tr>
      <td>59398</td>
      <td>31282</td>
      <td>0.0</td>
      <td>2011-03-08</td>
      <td>Malec</td>
      <td>0</td>
      <td>Musa</td>
      <td>35.861315</td>
      <td>-6.378573</td>
      <td>Mshoro</td>
      <td>0</td>
      <td>Rufiji</td>
      <td>Mwinyi</td>
      <td>Dodoma</td>
      <td>1</td>
      <td>4</td>
      <td>Chamwino</td>
      <td>Mvumi Makulu</td>
      <td>0</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>NaN</td>
      <td>True</td>
      <td>0</td>
      <td>nira/tanira</td>
      <td>nira/tanira</td>
      <td>handpump</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>insufficient</td>
      <td>insufficient</td>
      <td>shallow well</td>
      <td>shallow well</td>
      <td>groundwater</td>
      <td>hand pump</td>
      <td>hand pump</td>
    </tr>
    <tr>
      <td>59399</td>
      <td>26348</td>
      <td>0.0</td>
      <td>2011-03-23</td>
      <td>World Bank</td>
      <td>191</td>
      <td>World</td>
      <td>38.104048</td>
      <td>-6.747464</td>
      <td>Kwa Mzee Lugawa</td>
      <td>0</td>
      <td>Wami / Ruvu</td>
      <td>Kikatanyemba</td>
      <td>Morogoro</td>
      <td>5</td>
      <td>2</td>
      <td>Morogoro Rural</td>
      <td>Ngerengere</td>
      <td>150</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>NaN</td>
      <td>True</td>
      <td>2002</td>
      <td>nira/tanira</td>
      <td>nira/tanira</td>
      <td>handpump</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>pay when scheme fails</td>
      <td>on failure</td>
      <td>salty</td>
      <td>salty</td>
      <td>enough</td>
      <td>enough</td>
      <td>shallow well</td>
      <td>shallow well</td>
      <td>groundwater</td>
      <td>hand pump</td>
      <td>hand pump</td>
    </tr>
  </tbody>
</table>
<p>59400 rows × 40 columns</p>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>status_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>69572</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>1</td>
      <td>8776</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>2</td>
      <td>34310</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>3</td>
      <td>67743</td>
      <td>non functional</td>
    </tr>
    <tr>
      <td>4</td>
      <td>19728</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>59395</td>
      <td>60739</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>59396</td>
      <td>27263</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>59397</td>
      <td>37057</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>59398</td>
      <td>31282</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>59399</td>
      <td>26348</td>
      <td>functional</td>
    </tr>
  </tbody>
</table>
<p>59400 rows × 2 columns</p>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>amount_tsh</th>
      <th>date_recorded</th>
      <th>funder</th>
      <th>gps_height</th>
      <th>installer</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>wpt_name</th>
      <th>num_private</th>
      <th>basin</th>
      <th>subvillage</th>
      <th>region</th>
      <th>region_code</th>
      <th>district_code</th>
      <th>lga</th>
      <th>ward</th>
      <th>population</th>
      <th>public_meeting</th>
      <th>recorded_by</th>
      <th>scheme_management</th>
      <th>scheme_name</th>
      <th>permit</th>
      <th>construction_year</th>
      <th>extraction_type</th>
      <th>extraction_type_group</th>
      <th>extraction_type_class</th>
      <th>management</th>
      <th>management_group</th>
      <th>payment</th>
      <th>payment_type</th>
      <th>water_quality</th>
      <th>quality_group</th>
      <th>quantity</th>
      <th>quantity_group</th>
      <th>source</th>
      <th>source_type</th>
      <th>source_class</th>
      <th>waterpoint_type</th>
      <th>waterpoint_type_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>50785</td>
      <td>0.0</td>
      <td>2013-02-04</td>
      <td>Dmdd</td>
      <td>1996</td>
      <td>DMDD</td>
      <td>35.290799</td>
      <td>-4.059696</td>
      <td>Dinamu Secondary School</td>
      <td>0</td>
      <td>Internal</td>
      <td>Magoma</td>
      <td>Manyara</td>
      <td>21</td>
      <td>3</td>
      <td>Mbulu</td>
      <td>Bashay</td>
      <td>321</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>Parastatal</td>
      <td>NaN</td>
      <td>True</td>
      <td>2012</td>
      <td>other</td>
      <td>other</td>
      <td>other</td>
      <td>parastatal</td>
      <td>parastatal</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>seasonal</td>
      <td>seasonal</td>
      <td>rainwater harvesting</td>
      <td>rainwater harvesting</td>
      <td>surface</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <td>1</td>
      <td>51630</td>
      <td>0.0</td>
      <td>2013-02-04</td>
      <td>Government Of Tanzania</td>
      <td>1569</td>
      <td>DWE</td>
      <td>36.656709</td>
      <td>-3.309214</td>
      <td>Kimnyak</td>
      <td>0</td>
      <td>Pangani</td>
      <td>Kimnyak</td>
      <td>Arusha</td>
      <td>2</td>
      <td>2</td>
      <td>Arusha Rural</td>
      <td>Kimnyaki</td>
      <td>300</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>TPRI pipe line</td>
      <td>True</td>
      <td>2000</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>insufficient</td>
      <td>insufficient</td>
      <td>spring</td>
      <td>spring</td>
      <td>groundwater</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <td>2</td>
      <td>17168</td>
      <td>0.0</td>
      <td>2013-02-01</td>
      <td>NaN</td>
      <td>1567</td>
      <td>NaN</td>
      <td>34.767863</td>
      <td>-5.004344</td>
      <td>Puma Secondary</td>
      <td>0</td>
      <td>Internal</td>
      <td>Msatu</td>
      <td>Singida</td>
      <td>13</td>
      <td>2</td>
      <td>Singida Rural</td>
      <td>Puma</td>
      <td>500</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>P</td>
      <td>NaN</td>
      <td>2010</td>
      <td>other</td>
      <td>other</td>
      <td>other</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>insufficient</td>
      <td>insufficient</td>
      <td>rainwater harvesting</td>
      <td>rainwater harvesting</td>
      <td>surface</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <td>3</td>
      <td>45559</td>
      <td>0.0</td>
      <td>2013-01-22</td>
      <td>Finn Water</td>
      <td>267</td>
      <td>FINN WATER</td>
      <td>38.058046</td>
      <td>-9.418672</td>
      <td>Kwa Mzee Pange</td>
      <td>0</td>
      <td>Ruvuma / Southern Coast</td>
      <td>Kipindimbi</td>
      <td>Lindi</td>
      <td>80</td>
      <td>43</td>
      <td>Liwale</td>
      <td>Mkutano</td>
      <td>250</td>
      <td>NaN</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>NaN</td>
      <td>True</td>
      <td>1987</td>
      <td>other</td>
      <td>other</td>
      <td>other</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>unknown</td>
      <td>unknown</td>
      <td>soft</td>
      <td>good</td>
      <td>dry</td>
      <td>dry</td>
      <td>shallow well</td>
      <td>shallow well</td>
      <td>groundwater</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <td>4</td>
      <td>49871</td>
      <td>500.0</td>
      <td>2013-03-27</td>
      <td>Bruder</td>
      <td>1260</td>
      <td>BRUDER</td>
      <td>35.006123</td>
      <td>-10.950412</td>
      <td>Kwa Mzee Turuka</td>
      <td>0</td>
      <td>Ruvuma / Southern Coast</td>
      <td>Losonga</td>
      <td>Ruvuma</td>
      <td>10</td>
      <td>3</td>
      <td>Mbinga</td>
      <td>Mbinga Urban</td>
      <td>60</td>
      <td>NaN</td>
      <td>GeoData Consultants Ltd</td>
      <td>Water Board</td>
      <td>BRUDER</td>
      <td>True</td>
      <td>2000</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>water board</td>
      <td>user-group</td>
      <td>pay monthly</td>
      <td>monthly</td>
      <td>soft</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>spring</td>
      <td>spring</td>
      <td>groundwater</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>14845</td>
      <td>39307</td>
      <td>0.0</td>
      <td>2011-02-24</td>
      <td>Danida</td>
      <td>34</td>
      <td>Da</td>
      <td>38.852669</td>
      <td>-6.582841</td>
      <td>Kwambwezi</td>
      <td>0</td>
      <td>Wami / Ruvu</td>
      <td>Yombo</td>
      <td>Pwani</td>
      <td>6</td>
      <td>1</td>
      <td>Bagamoyo</td>
      <td>Yombo</td>
      <td>20</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>Bagamoyo wate</td>
      <td>True</td>
      <td>1988</td>
      <td>mono</td>
      <td>mono</td>
      <td>motorpump</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>river</td>
      <td>river/lake</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <td>14846</td>
      <td>18990</td>
      <td>1000.0</td>
      <td>2011-03-21</td>
      <td>Hiap</td>
      <td>0</td>
      <td>HIAP</td>
      <td>37.451633</td>
      <td>-5.350428</td>
      <td>Bonde La Mkondoa</td>
      <td>0</td>
      <td>Pangani</td>
      <td>Mkondoa</td>
      <td>Tanga</td>
      <td>4</td>
      <td>7</td>
      <td>Kilindi</td>
      <td>Mvungwe</td>
      <td>2960</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>NaN</td>
      <td>False</td>
      <td>1994</td>
      <td>nira/tanira</td>
      <td>nira/tanira</td>
      <td>handpump</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>pay annually</td>
      <td>annually</td>
      <td>salty</td>
      <td>salty</td>
      <td>insufficient</td>
      <td>insufficient</td>
      <td>shallow well</td>
      <td>shallow well</td>
      <td>groundwater</td>
      <td>hand pump</td>
      <td>hand pump</td>
    </tr>
    <tr>
      <td>14847</td>
      <td>28749</td>
      <td>0.0</td>
      <td>2013-03-04</td>
      <td>NaN</td>
      <td>1476</td>
      <td>NaN</td>
      <td>34.739804</td>
      <td>-4.585587</td>
      <td>Bwawani</td>
      <td>0</td>
      <td>Internal</td>
      <td>Juhudi</td>
      <td>Singida</td>
      <td>13</td>
      <td>2</td>
      <td>Singida Rural</td>
      <td>Ughandi</td>
      <td>200</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2010</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>insufficient</td>
      <td>insufficient</td>
      <td>dam</td>
      <td>dam</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <td>14848</td>
      <td>33492</td>
      <td>0.0</td>
      <td>2013-02-18</td>
      <td>Germany</td>
      <td>998</td>
      <td>DWE</td>
      <td>35.432732</td>
      <td>-10.584159</td>
      <td>Kwa John</td>
      <td>0</td>
      <td>Lake Nyasa</td>
      <td>Namakinga B</td>
      <td>Ruvuma</td>
      <td>10</td>
      <td>2</td>
      <td>Songea Rural</td>
      <td>Maposeni</td>
      <td>150</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>Mradi wa maji wa maposeni</td>
      <td>True</td>
      <td>2009</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>insufficient</td>
      <td>insufficient</td>
      <td>river</td>
      <td>river/lake</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <td>14849</td>
      <td>68707</td>
      <td>0.0</td>
      <td>2013-02-13</td>
      <td>Government Of Tanzania</td>
      <td>481</td>
      <td>Government</td>
      <td>34.765054</td>
      <td>-11.226012</td>
      <td>Kwa Mzee Chagala</td>
      <td>0</td>
      <td>Lake Nyasa</td>
      <td>Kamba</td>
      <td>Ruvuma</td>
      <td>10</td>
      <td>3</td>
      <td>Mbinga</td>
      <td>Mbamba bay</td>
      <td>40</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>DANIDA</td>
      <td>True</td>
      <td>2008</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>dry</td>
      <td>dry</td>
      <td>spring</td>
      <td>spring</td>
      <td>groundwater</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
    </tr>
  </tbody>
</table>
<p>14850 rows × 40 columns</p>
</div>



```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>amount_tsh</th>
      <th>gps_height</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>num_private</th>
      <th>region_code</th>
      <th>district_code</th>
      <th>population</th>
      <th>construction_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>59400.000000</td>
      <td>59400.000000</td>
      <td>59400.000000</td>
      <td>59400.000000</td>
      <td>5.940000e+04</td>
      <td>59400.000000</td>
      <td>59400.000000</td>
      <td>59400.000000</td>
      <td>59400.000000</td>
      <td>59400.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>37115.131768</td>
      <td>317.650385</td>
      <td>668.297239</td>
      <td>34.077427</td>
      <td>-5.706033e+00</td>
      <td>0.474141</td>
      <td>15.297003</td>
      <td>5.629747</td>
      <td>179.909983</td>
      <td>1300.652475</td>
    </tr>
    <tr>
      <td>std</td>
      <td>21453.128371</td>
      <td>2997.574558</td>
      <td>693.116350</td>
      <td>6.567432</td>
      <td>2.946019e+00</td>
      <td>12.236230</td>
      <td>17.587406</td>
      <td>9.633649</td>
      <td>471.482176</td>
      <td>951.620547</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-90.000000</td>
      <td>0.000000</td>
      <td>-1.164944e+01</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>18519.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>33.090347</td>
      <td>-8.540621e+00</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>37061.500000</td>
      <td>0.000000</td>
      <td>369.000000</td>
      <td>34.908743</td>
      <td>-5.021597e+00</td>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>3.000000</td>
      <td>25.000000</td>
      <td>1986.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>55656.500000</td>
      <td>20.000000</td>
      <td>1319.250000</td>
      <td>37.178387</td>
      <td>-3.326156e+00</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>5.000000</td>
      <td>215.000000</td>
      <td>2004.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>74247.000000</td>
      <td>350000.000000</td>
      <td>2770.000000</td>
      <td>40.345193</td>
      <td>-2.000000e-08</td>
      <td>1776.000000</td>
      <td>99.000000</td>
      <td>80.000000</td>
      <td>30500.000000</td>
      <td>2013.000000</td>
    </tr>
  </tbody>
</table>
</div>



### add labels to main df


```python
df['status_group'] = training_labels['status_group']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>amount_tsh</th>
      <th>date_recorded</th>
      <th>funder</th>
      <th>gps_height</th>
      <th>installer</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>wpt_name</th>
      <th>num_private</th>
      <th>basin</th>
      <th>subvillage</th>
      <th>region</th>
      <th>region_code</th>
      <th>district_code</th>
      <th>lga</th>
      <th>ward</th>
      <th>population</th>
      <th>public_meeting</th>
      <th>recorded_by</th>
      <th>scheme_management</th>
      <th>scheme_name</th>
      <th>permit</th>
      <th>construction_year</th>
      <th>extraction_type</th>
      <th>extraction_type_group</th>
      <th>extraction_type_class</th>
      <th>management</th>
      <th>management_group</th>
      <th>payment</th>
      <th>payment_type</th>
      <th>water_quality</th>
      <th>quality_group</th>
      <th>quantity</th>
      <th>quantity_group</th>
      <th>source</th>
      <th>source_type</th>
      <th>source_class</th>
      <th>waterpoint_type</th>
      <th>waterpoint_type_group</th>
      <th>status_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>69572</td>
      <td>6000.0</td>
      <td>2011-03-14</td>
      <td>Roman</td>
      <td>1390</td>
      <td>Roman</td>
      <td>34.938093</td>
      <td>-9.856322</td>
      <td>none</td>
      <td>0</td>
      <td>Lake Nyasa</td>
      <td>Mnyusi B</td>
      <td>Iringa</td>
      <td>11</td>
      <td>5</td>
      <td>Ludewa</td>
      <td>Mundindi</td>
      <td>109</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>Roman</td>
      <td>False</td>
      <td>1999</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>pay annually</td>
      <td>annually</td>
      <td>soft</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>spring</td>
      <td>spring</td>
      <td>groundwater</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>1</td>
      <td>8776</td>
      <td>0.0</td>
      <td>2013-03-06</td>
      <td>Grumeti</td>
      <td>1399</td>
      <td>GRUMETI</td>
      <td>34.698766</td>
      <td>-2.147466</td>
      <td>Zahanati</td>
      <td>0</td>
      <td>Lake Victoria</td>
      <td>Nyamara</td>
      <td>Mara</td>
      <td>20</td>
      <td>2</td>
      <td>Serengeti</td>
      <td>Natta</td>
      <td>280</td>
      <td>NaN</td>
      <td>GeoData Consultants Ltd</td>
      <td>Other</td>
      <td>NaN</td>
      <td>True</td>
      <td>2010</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>wug</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>insufficient</td>
      <td>insufficient</td>
      <td>rainwater harvesting</td>
      <td>rainwater harvesting</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>2</td>
      <td>34310</td>
      <td>25.0</td>
      <td>2013-02-25</td>
      <td>Lottery Club</td>
      <td>686</td>
      <td>World vision</td>
      <td>37.460664</td>
      <td>-3.821329</td>
      <td>Kwa Mahundi</td>
      <td>0</td>
      <td>Pangani</td>
      <td>Majengo</td>
      <td>Manyara</td>
      <td>21</td>
      <td>4</td>
      <td>Simanjiro</td>
      <td>Ngorika</td>
      <td>250</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>Nyumba ya mungu pipe scheme</td>
      <td>True</td>
      <td>2009</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>pay per bucket</td>
      <td>per bucket</td>
      <td>soft</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>dam</td>
      <td>dam</td>
      <td>surface</td>
      <td>communal standpipe multiple</td>
      <td>communal standpipe</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>3</td>
      <td>67743</td>
      <td>0.0</td>
      <td>2013-01-28</td>
      <td>Unicef</td>
      <td>263</td>
      <td>UNICEF</td>
      <td>38.486161</td>
      <td>-11.155298</td>
      <td>Zahanati Ya Nanyumbu</td>
      <td>0</td>
      <td>Ruvuma / Southern Coast</td>
      <td>Mahakamani</td>
      <td>Mtwara</td>
      <td>90</td>
      <td>63</td>
      <td>Nanyumbu</td>
      <td>Nanyumbu</td>
      <td>58</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>VWC</td>
      <td>NaN</td>
      <td>True</td>
      <td>1986</td>
      <td>submersible</td>
      <td>submersible</td>
      <td>submersible</td>
      <td>vwc</td>
      <td>user-group</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>dry</td>
      <td>dry</td>
      <td>machine dbh</td>
      <td>borehole</td>
      <td>groundwater</td>
      <td>communal standpipe multiple</td>
      <td>communal standpipe</td>
      <td>non functional</td>
    </tr>
    <tr>
      <td>4</td>
      <td>19728</td>
      <td>0.0</td>
      <td>2011-07-13</td>
      <td>Action In A</td>
      <td>0</td>
      <td>Artisan</td>
      <td>31.130847</td>
      <td>-1.825359</td>
      <td>Shuleni</td>
      <td>0</td>
      <td>Lake Victoria</td>
      <td>Kyanyamisa</td>
      <td>Kagera</td>
      <td>18</td>
      <td>1</td>
      <td>Karagwe</td>
      <td>Nyakasimbi</td>
      <td>0</td>
      <td>True</td>
      <td>GeoData Consultants Ltd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>0</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>gravity</td>
      <td>other</td>
      <td>other</td>
      <td>never pay</td>
      <td>never pay</td>
      <td>soft</td>
      <td>good</td>
      <td>seasonal</td>
      <td>seasonal</td>
      <td>rainwater harvesting</td>
      <td>rainwater harvesting</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
      <td>functional</td>
    </tr>
  </tbody>
</table>
</div>



## Check nulls

Notice there are similar nulls in funder and installer cols.


```python
## Check null values
import missingno
missingno.matrix(df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c1fd342e8>




![png](output_26_1.png)



```python
df.isna().sum()
```




    id                           0
    amount_tsh                   0
    date_recorded                0
    funder                    3635
    gps_height                   0
    installer                 3655
    longitude                    0
    latitude                     0
    wpt_name                     0
    num_private                  0
    basin                        0
    subvillage                 371
    region                       0
    region_code                  0
    district_code                0
    lga                          0
    ward                         0
    population                   0
    public_meeting            3334
    recorded_by                  0
    scheme_management         3877
    scheme_name              28166
    permit                    3056
    construction_year            0
    extraction_type              0
    extraction_type_group        0
    extraction_type_class        0
    management                   0
    management_group             0
    payment                      0
    payment_type                 0
    water_quality                0
    quality_group                0
    quantity                     0
    quantity_group               0
    source                       0
    source_type                  0
    source_class                 0
    waterpoint_type              0
    waterpoint_type_group        0
    status_group                 0
    dtype: int64



Class imbalance will need to be addressed.


```python
df['status_group'].value_counts(normalize=True)
```




    functional                 0.543081
    non functional             0.384242
    functional needs repair    0.072677
    Name: status_group, dtype: float64



## drop null val cols


```python
# """
# cols to delete:
# id ** - don't need, will match with index
# date_recorded - don't need, can't do time series
# funder - Who funded the well - null vals
# installer - Organization that installed the well ** - null vals
# subvillage - Geographic location ** - don't need, cat, using region code, nulls
# public_meeting - True/False - don't need, null vals
# scheme_management - Who operates the waterpoint ++ - nulls, don't need
# scheme_name - Who operates the waterpoint ++ ** - nulls, don't need
# permit - If the waterpoint is permitted - nulls, don't need
# """
```


```python
cols_to_drop = ['id', 'date_recorded', 'funder',
       'installer', 'subvillage', 'public_meeting',
       'scheme_management', 'scheme_name', 'permit']
                
                
df.drop(columns = cols_to_drop ,axis=1, inplace=True)
df.columns
```




    Index(['amount_tsh', 'gps_height', 'longitude', 'latitude', 'wpt_name',
           'num_private', 'basin', 'region', 'region_code', 'district_code', 'lga',
           'ward', 'population', 'recorded_by', 'construction_year',
           'extraction_type', 'extraction_type_group', 'extraction_type_class',
           'management', 'management_group', 'payment', 'payment_type',
           'water_quality', 'quality_group', 'quantity', 'quantity_group',
           'source', 'source_type', 'source_class', 'waterpoint_type',
           'waterpoint_type_group', 'status_group'],
          dtype='object')




```python
df.isna().sum()
```




    amount_tsh               0
    gps_height               0
    longitude                0
    latitude                 0
    wpt_name                 0
    num_private              0
    basin                    0
    region                   0
    region_code              0
    district_code            0
    lga                      0
    ward                     0
    population               0
    recorded_by              0
    construction_year        0
    extraction_type          0
    extraction_type_group    0
    extraction_type_class    0
    management               0
    management_group         0
    payment                  0
    payment_type             0
    water_quality            0
    quality_group            0
    quantity                 0
    quantity_group           0
    source                   0
    source_type              0
    source_class             0
    waterpoint_type          0
    waterpoint_type_group    0
    status_group             0
    dtype: int64




```python
df.dtypes
```




    amount_tsh               float64
    gps_height                 int64
    longitude                float64
    latitude                 float64
    wpt_name                  object
    num_private                int64
    basin                     object
    region                    object
    region_code                int64
    district_code              int64
    lga                       object
    ward                      object
    population                 int64
    recorded_by               object
    construction_year          int64
    extraction_type           object
    extraction_type_group     object
    extraction_type_class     object
    management                object
    management_group          object
    payment                   object
    payment_type              object
    water_quality             object
    quality_group             object
    quantity                  object
    quantity_group            object
    source                    object
    source_type               object
    source_class              object
    waterpoint_type           object
    waterpoint_type_group     object
    status_group              object
    dtype: object



# SCRUB

## checking for weird values


```python
for col in df.columns:
    print(df[col].value_counts())
```

    0.0         41639
    500.0        3102
    50.0         2472
    1000.0       1488
    20.0         1463
                ...  
    8500.0          1
    6300.0          1
    220.0           1
    138000.0        1
    12.0            1
    Name: amount_tsh, Length: 98, dtype: int64
     0       20438
    -15         60
    -16         55
    -13         55
    -20         52
             ...  
     2285        1
     2424        1
     2552        1
     2413        1
     2385        1
    Name: gps_height, Length: 2428, dtype: int64
    0.000000     1812
    37.540901       2
    33.010510       2
    39.093484       2
    32.972719       2
                 ... 
    37.579803       1
    33.196490       1
    34.017119       1
    33.788326       1
    30.163579       1
    Name: longitude, Length: 57516, dtype: int64
    -2.000000e-08    1812
    -6.985842e+00       2
    -3.797579e+00       2
    -6.981884e+00       2
    -7.104625e+00       2
                     ... 
    -5.726001e+00       1
    -9.646831e+00       1
    -8.124530e+00       1
    -2.535985e+00       1
    -2.598965e+00       1
    Name: latitude, Length: 57517, dtype: int64
    none                3563
    Shuleni             1748
    Zahanati             830
    Msikitini            535
    Kanisani             323
                        ... 
    Msemwa                 1
    Kwa Kitweko            1
    Kwa Sakina Amiri       1
    Kwa Chide Wolde        1
    Kwa Mzee Karoli        1
    Name: wpt_name, Length: 37400, dtype: int64
    0      58643
    6         81
    1         73
    5         46
    8         46
           ...  
    180        1
    213        1
    23         1
    55         1
    94         1
    Name: num_private, Length: 65, dtype: int64
    Lake Victoria              10248
    Pangani                     8940
    Rufiji                      7976
    Internal                    7785
    Lake Tanganyika             6432
    Wami / Ruvu                 5987
    Lake Nyasa                  5085
    Ruvuma / Southern Coast     4493
    Lake Rukwa                  2454
    Name: basin, dtype: int64
    Iringa           5294
    Shinyanga        4982
    Mbeya            4639
    Kilimanjaro      4379
    Morogoro         4006
    Arusha           3350
    Kagera           3316
    Mwanza           3102
    Kigoma           2816
    Ruvuma           2640
    Pwani            2635
    Tanga            2547
    Dodoma           2201
    Singida          2093
    Mara             1969
    Tabora           1959
    Rukwa            1808
    Mtwara           1730
    Manyara          1583
    Lindi            1546
    Dar es Salaam     805
    Name: region, dtype: int64
    11    5300
    17    5011
    12    4639
    3     4379
    5     4040
    18    3324
    19    3047
    2     3024
    16    2816
    10    2640
    4     2513
    1     2201
    13    2093
    14    1979
    20    1969
    15    1808
    6     1609
    21    1583
    80    1238
    60    1025
    90     917
    7      805
    99     423
    9      390
    24     326
    8      300
    40       1
    Name: region_code, dtype: int64
    1     12203
    2     11173
    3      9998
    4      8999
    5      4356
    6      4074
    7      3343
    8      1043
    30      995
    33      874
    53      745
    43      505
    13      391
    23      293
    63      195
    62      109
    60       63
    0        23
    80       12
    67        6
    Name: district_code, dtype: int64
    Njombe          2503
    Arusha Rural    1252
    Moshi Rural     1251
    Bariadi         1177
    Rungwe          1106
                    ... 
    Moshi Urban       79
    Kigoma Urban      71
    Arusha Urban      63
    Lindi Urban       21
    Nyamagana          1
    Name: lga, Length: 125, dtype: int64
    Igosi        307
    Imalinyi     252
    Siha Kati    232
    Mdandu       231
    Nduruma      217
                ... 
    Mkumbi         1
    Igogo          1
    Kapilula       1
    Themi          1
    Burungura      1
    Name: ward, Length: 2092, dtype: int64
    0       21381
    1        7025
    200      1940
    150      1892
    250      1681
            ...  
    3241        1
    1960        1
    1685        1
    2248        1
    1439        1
    Name: population, Length: 1049, dtype: int64
    GeoData Consultants Ltd    59400
    Name: recorded_by, dtype: int64
    0       20709
    2010     2645
    2008     2613
    2009     2533
    2000     2091
    2007     1587
    2006     1471
    2003     1286
    2011     1256
    2004     1123
    2012     1084
    2002     1075
    1978     1037
    1995     1014
    2005     1011
    1999      979
    1998      966
    1990      954
    1985      945
    1980      811
    1996      811
    1984      779
    1982      744
    1994      738
    1972      708
    1974      676
    1997      644
    1992      640
    1993      608
    2001      540
    1988      521
    1983      488
    1975      437
    1986      434
    1976      414
    1970      411
    1991      324
    1989      316
    1987      302
    1981      238
    1977      202
    1979      192
    1973      184
    2013      176
    1971      145
    1960      102
    1967       88
    1963       85
    1968       77
    1969       59
    1964       40
    1962       30
    1961       21
    1965       19
    1966       17
    Name: construction_year, dtype: int64
    gravity                      26780
    nira/tanira                   8154
    other                         6430
    submersible                   4764
    swn 80                        3670
    mono                          2865
    india mark ii                 2400
    afridev                       1770
    ksb                           1415
    other - rope pump              451
    other - swn 81                 229
    windmill                       117
    india mark iii                  98
    cemo                            90
    other - play pump               85
    walimi                          48
    climax                          32
    other - mkulima/shinyanga        2
    Name: extraction_type, dtype: int64
    gravity            26780
    nira/tanira         8154
    other               6430
    submersible         6179
    swn 80              3670
    mono                2865
    india mark ii       2400
    afridev             1770
    rope pump            451
    other handpump       364
    other motorpump      122
    wind-powered         117
    india mark iii        98
    Name: extraction_type_group, dtype: int64
    gravity         26780
    handpump        16456
    other            6430
    submersible      6179
    motorpump        2987
    rope pump         451
    wind-powered      117
    Name: extraction_type_class, dtype: int64
    vwc                 40507
    wug                  6515
    water board          2933
    wua                  2535
    private operator     1971
    parastatal           1768
    water authority       904
    other                 844
    company               685
    unknown               561
    other - school         99
    trust                  78
    Name: management, dtype: int64
    user-group    52490
    commercial     3638
    parastatal     1768
    other           943
    unknown         561
    Name: management_group, dtype: int64
    never pay                25348
    pay per bucket            8985
    pay monthly               8300
    unknown                   8157
    pay when scheme fails     3914
    pay annually              3642
    other                     1054
    Name: payment, dtype: int64
    never pay     25348
    per bucket     8985
    monthly        8300
    unknown        8157
    on failure     3914
    annually       3642
    other          1054
    Name: payment_type, dtype: int64
    soft                  50818
    salty                  4856
    unknown                1876
    milky                   804
    coloured                490
    salty abandoned         339
    fluoride                200
    fluoride abandoned       17
    Name: water_quality, dtype: int64
    good        50818
    salty        5195
    unknown      1876
    milky         804
    colored       490
    fluoride      217
    Name: quality_group, dtype: int64
    enough          33186
    insufficient    15129
    dry              6246
    seasonal         4050
    unknown           789
    Name: quantity, dtype: int64
    enough          33186
    insufficient    15129
    dry              6246
    seasonal         4050
    unknown           789
    Name: quantity_group, dtype: int64
    spring                  17021
    shallow well            16824
    machine dbh             11075
    river                    9612
    rainwater harvesting     2295
    hand dtw                  874
    lake                      765
    dam                       656
    other                     212
    unknown                    66
    Name: source, dtype: int64
    spring                  17021
    shallow well            16824
    borehole                11949
    river/lake              10377
    rainwater harvesting     2295
    dam                       656
    other                     278
    Name: source_type, dtype: int64
    groundwater    45794
    surface        13328
    unknown          278
    Name: source_class, dtype: int64
    communal standpipe             28522
    hand pump                      17488
    other                           6380
    communal standpipe multiple     6103
    improved spring                  784
    cattle trough                    116
    dam                                7
    Name: waterpoint_type, dtype: int64
    communal standpipe    34625
    hand pump             17488
    other                  6380
    improved spring         784
    cattle trough           116
    dam                       7
    Name: waterpoint_type_group, dtype: int64
    functional                 32259
    non functional             22824
    functional needs repair     4317
    Name: status_group, dtype: int64



```python
# tsh has a lot of 0s, could be related to non functional pumps
# lot of 0s in gps height, could be at sea level because tanzania is on the ocean
# 1812 0s in longitude, maybe delete those rows, it's only a small part of the total 59,400 rows - 3%
# 1812 of latitude -2.000000e -08, delete these rows bc they're the same as the 0 degree longitude ones
# population has lot of 0 or as population 1, is that right? could be that it's not local to 
    # a population - I'm going to leave it for now
# theres a lot of 0s in the construction year column - should i change the zeros to unknown and use cat, or fil with 1900 and use as numeric
# might not need both source type and water point type group but i'll keep both for now
```


```python
df.loc[df['amount_tsh']==0]['status_group'].value_counts()
```




    functional                 19706
    non functional             18885
    functional needs repair     3048
    Name: status_group, dtype: int64




```python
# there are many pumps still functioning but still have 0 amount_tsh. keep colum for now with 0s listed
```

## Feature selection - drop useless columns

Don't need:
- Wpt_name - it's just name of the pump, not helpful. 
- Num_private - no description for this column. it's mostly 0s
- Recorded_by - it's a constant. useless.


```python
cols_to_drop = ['wpt_name', 'num_private', 'recorded_by']
                
df.drop(columns = cols_to_drop ,axis=1, inplace=True)
df.columns
```




    Index(['amount_tsh', 'gps_height', 'longitude', 'latitude', 'basin', 'region',
           'region_code', 'district_code', 'lga', 'ward', 'population',
           'construction_year', 'extraction_type', 'extraction_type_group',
           'extraction_type_class', 'management', 'management_group', 'payment',
           'payment_type', 'water_quality', 'quality_group', 'quantity',
           'quantity_group', 'source', 'source_type', 'source_class',
           'waterpoint_type', 'waterpoint_type_group', 'status_group'],
          dtype='object')



### drop 0 deg long and -2e-8 lat rows 


```python
# drop rows in long/lat that have 0s, because these coordinates are fillers that place them in the
# the atlantic ocean on the opposite side of the continent
# in future explorations, could fill data with woth coordinates based on specific region
# when have more time
```


```python
df = df[(df['longitude'] != 0) & (df['latitude'] != -2.000000e-08)]
```


```python
df.shape
```




    (57588, 29)



# EXPLORE

## check for categoricals


```python
pd.plotting.scatter_matrix(df, figsize=(10,10))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x1c2357dc18>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c230b7320>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c231fac50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c1f7c2f60>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c22aa30f0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c2299c1d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c2320e668>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c1e59fa58>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1c1e59fa90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c22915a58>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c22e52b70>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c22ff3160>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1818a54710>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1818a55cc0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1818ba32b0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1818b59860>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1c23545390>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c2302c400>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1818c179b0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c1d51df60>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c2345d550>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c23490b00>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c22f2d0f0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c22f5f6a0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1c2288fc50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c2284c240>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1818ac47f0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1818af7da0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c212c3390>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10ecb5940>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1818a76ef0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1818b144e0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1818b42a20>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1818c54fd0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1a1a53f5c0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c1d2fab70>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c1d338160>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c1d366710>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c1d398cc0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c1d47d2b0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1c1d4ae860>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c1e55de10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c1e5da400>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c1e8569b0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c1e888f60>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c1faf3550>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c205f5b00>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c206350f0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1c206636a0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c20696c50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c206d5240>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c207047f0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c20735da0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c20773390>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c207a4940>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c207d7ef0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1c208144e0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c20843a90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c20882080>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c208b2630>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c208e4be0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c209221d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c20950780>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1c20983d30>]],
          dtype=object)




![png](output_50_1.png)



```python
df.dtypes
```




    amount_tsh               float64
    gps_height                 int64
    longitude                float64
    latitude                 float64
    basin                     object
    region                    object
    region_code                int64
    district_code              int64
    lga                       object
    ward                      object
    population                 int64
    construction_year          int64
    extraction_type           object
    extraction_type_group     object
    extraction_type_class     object
    management                object
    management_group          object
    payment                   object
    payment_type              object
    water_quality             object
    quality_group             object
    quantity                  object
    quantity_group            object
    source                    object
    source_type               object
    source_class              object
    waterpoint_type           object
    waterpoint_type_group     object
    status_group              object
    dtype: object




```python
# boatloads of categoricals
```

## split x and y dfs


```python
df['status_group']
```




    0            functional
    1            functional
    2            functional
    3        non functional
    4            functional
                  ...      
    59395        functional
    59396        functional
    59397        functional
    59398        functional
    59399        functional
    Name: status_group, Length: 57588, dtype: object




```python
y = df['status_group']
X = df.drop('status_group', axis=1)
```

## one hot cats


```python
X = pd.get_dummies(X)
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount_tsh</th>
      <th>gps_height</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>region_code</th>
      <th>district_code</th>
      <th>population</th>
      <th>construction_year</th>
      <th>basin_Internal</th>
      <th>basin_Lake Nyasa</th>
      <th>basin_Lake Rukwa</th>
      <th>basin_Lake Tanganyika</th>
      <th>basin_Lake Victoria</th>
      <th>basin_Pangani</th>
      <th>basin_Rufiji</th>
      <th>basin_Ruvuma / Southern Coast</th>
      <th>basin_Wami / Ruvu</th>
      <th>region_Arusha</th>
      <th>region_Dar es Salaam</th>
      <th>region_Dodoma</th>
      <th>region_Iringa</th>
      <th>region_Kagera</th>
      <th>region_Kigoma</th>
      <th>region_Kilimanjaro</th>
      <th>region_Lindi</th>
      <th>region_Manyara</th>
      <th>region_Mara</th>
      <th>region_Mbeya</th>
      <th>region_Morogoro</th>
      <th>region_Mtwara</th>
      <th>region_Mwanza</th>
      <th>region_Pwani</th>
      <th>region_Rukwa</th>
      <th>region_Ruvuma</th>
      <th>region_Shinyanga</th>
      <th>region_Singida</th>
      <th>region_Tabora</th>
      <th>region_Tanga</th>
      <th>lga_Arusha Rural</th>
      <th>lga_Arusha Urban</th>
      <th>...</th>
      <th>quantity_seasonal</th>
      <th>quantity_unknown</th>
      <th>quantity_group_dry</th>
      <th>quantity_group_enough</th>
      <th>quantity_group_insufficient</th>
      <th>quantity_group_seasonal</th>
      <th>quantity_group_unknown</th>
      <th>source_dam</th>
      <th>source_hand dtw</th>
      <th>source_lake</th>
      <th>source_machine dbh</th>
      <th>source_other</th>
      <th>source_rainwater harvesting</th>
      <th>source_river</th>
      <th>source_shallow well</th>
      <th>source_spring</th>
      <th>source_unknown</th>
      <th>source_type_borehole</th>
      <th>source_type_dam</th>
      <th>source_type_other</th>
      <th>source_type_rainwater harvesting</th>
      <th>source_type_river/lake</th>
      <th>source_type_shallow well</th>
      <th>source_type_spring</th>
      <th>source_class_groundwater</th>
      <th>source_class_surface</th>
      <th>source_class_unknown</th>
      <th>waterpoint_type_cattle trough</th>
      <th>waterpoint_type_communal standpipe</th>
      <th>waterpoint_type_communal standpipe multiple</th>
      <th>waterpoint_type_dam</th>
      <th>waterpoint_type_hand pump</th>
      <th>waterpoint_type_improved spring</th>
      <th>waterpoint_type_other</th>
      <th>waterpoint_type_group_cattle trough</th>
      <th>waterpoint_type_group_communal standpipe</th>
      <th>waterpoint_type_group_dam</th>
      <th>waterpoint_type_group_hand pump</th>
      <th>waterpoint_type_group_improved spring</th>
      <th>waterpoint_type_group_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>6000.0</td>
      <td>1390</td>
      <td>34.938093</td>
      <td>-9.856322</td>
      <td>11</td>
      <td>5</td>
      <td>109</td>
      <td>1999</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>1399</td>
      <td>34.698766</td>
      <td>-2.147466</td>
      <td>20</td>
      <td>2</td>
      <td>280</td>
      <td>2010</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>25.0</td>
      <td>686</td>
      <td>37.460664</td>
      <td>-3.821329</td>
      <td>21</td>
      <td>4</td>
      <td>250</td>
      <td>2009</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.0</td>
      <td>263</td>
      <td>38.486161</td>
      <td>-11.155298</td>
      <td>90</td>
      <td>63</td>
      <td>58</td>
      <td>1986</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>31.130847</td>
      <td>-1.825359</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2321 columns</p>
</div>




```python
# that's a crap load of columns. I'm going to drop some repeats because some of them are repeating information. 
```

## Feature selection - drop more columns that have repeat info


```python
df.columns
```




    Index(['amount_tsh', 'gps_height', 'longitude', 'latitude', 'basin', 'region',
           'region_code', 'district_code', 'lga', 'ward', 'population',
           'construction_year', 'extraction_type', 'extraction_type_group',
           'extraction_type_class', 'management', 'management_group', 'payment',
           'payment_type', 'water_quality', 'quality_group', 'quantity',
           'quantity_group', 'source', 'source_type', 'source_class',
           'waterpoint_type', 'waterpoint_type_group', 'status_group'],
          dtype='object')




```python
# """
# cols to delete:
# ward - Geographic location ** - don't need, has a million cats which = million columns


# """
```


```python
cols_to_drop = ['ward']
            # keep these others for now    
                #'extraction_type', 'extraction_type_group','management', 'payment',
       #'water_quality', 'quantity', 'source', 'source_class', 'waterpoint_type']
                
                
df.drop(columns = cols_to_drop ,axis=1, inplace=True)
df.columns
```




    Index(['amount_tsh', 'gps_height', 'longitude', 'latitude', 'basin', 'region',
           'region_code', 'district_code', 'lga', 'population',
           'construction_year', 'extraction_type', 'extraction_type_group',
           'extraction_type_class', 'management', 'management_group', 'payment',
           'payment_type', 'water_quality', 'quality_group', 'quantity',
           'quantity_group', 'source', 'source_type', 'source_class',
           'waterpoint_type', 'waterpoint_type_group', 'status_group'],
          dtype='object')



## one hot cats


```python
y = df['status_group']
X = df.drop('status_group', axis=1)
```


```python
X = pd.get_dummies(X)
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount_tsh</th>
      <th>gps_height</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>region_code</th>
      <th>district_code</th>
      <th>population</th>
      <th>construction_year</th>
      <th>basin_Internal</th>
      <th>basin_Lake Nyasa</th>
      <th>basin_Lake Rukwa</th>
      <th>basin_Lake Tanganyika</th>
      <th>basin_Lake Victoria</th>
      <th>basin_Pangani</th>
      <th>basin_Rufiji</th>
      <th>basin_Ruvuma / Southern Coast</th>
      <th>basin_Wami / Ruvu</th>
      <th>region_Arusha</th>
      <th>region_Dar es Salaam</th>
      <th>region_Dodoma</th>
      <th>region_Iringa</th>
      <th>region_Kagera</th>
      <th>region_Kigoma</th>
      <th>region_Kilimanjaro</th>
      <th>region_Lindi</th>
      <th>region_Manyara</th>
      <th>region_Mara</th>
      <th>region_Mbeya</th>
      <th>region_Morogoro</th>
      <th>region_Mtwara</th>
      <th>region_Mwanza</th>
      <th>region_Pwani</th>
      <th>region_Rukwa</th>
      <th>region_Ruvuma</th>
      <th>region_Shinyanga</th>
      <th>region_Singida</th>
      <th>region_Tabora</th>
      <th>region_Tanga</th>
      <th>lga_Arusha Rural</th>
      <th>lga_Arusha Urban</th>
      <th>...</th>
      <th>quantity_seasonal</th>
      <th>quantity_unknown</th>
      <th>quantity_group_dry</th>
      <th>quantity_group_enough</th>
      <th>quantity_group_insufficient</th>
      <th>quantity_group_seasonal</th>
      <th>quantity_group_unknown</th>
      <th>source_dam</th>
      <th>source_hand dtw</th>
      <th>source_lake</th>
      <th>source_machine dbh</th>
      <th>source_other</th>
      <th>source_rainwater harvesting</th>
      <th>source_river</th>
      <th>source_shallow well</th>
      <th>source_spring</th>
      <th>source_unknown</th>
      <th>source_type_borehole</th>
      <th>source_type_dam</th>
      <th>source_type_other</th>
      <th>source_type_rainwater harvesting</th>
      <th>source_type_river/lake</th>
      <th>source_type_shallow well</th>
      <th>source_type_spring</th>
      <th>source_class_groundwater</th>
      <th>source_class_surface</th>
      <th>source_class_unknown</th>
      <th>waterpoint_type_cattle trough</th>
      <th>waterpoint_type_communal standpipe</th>
      <th>waterpoint_type_communal standpipe multiple</th>
      <th>waterpoint_type_dam</th>
      <th>waterpoint_type_hand pump</th>
      <th>waterpoint_type_improved spring</th>
      <th>waterpoint_type_other</th>
      <th>waterpoint_type_group_cattle trough</th>
      <th>waterpoint_type_group_communal standpipe</th>
      <th>waterpoint_type_group_dam</th>
      <th>waterpoint_type_group_hand pump</th>
      <th>waterpoint_type_group_improved spring</th>
      <th>waterpoint_type_group_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>6000.0</td>
      <td>1390</td>
      <td>34.938093</td>
      <td>-9.856322</td>
      <td>11</td>
      <td>5</td>
      <td>109</td>
      <td>1999</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>1399</td>
      <td>34.698766</td>
      <td>-2.147466</td>
      <td>20</td>
      <td>2</td>
      <td>280</td>
      <td>2010</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>25.0</td>
      <td>686</td>
      <td>37.460664</td>
      <td>-3.821329</td>
      <td>21</td>
      <td>4</td>
      <td>250</td>
      <td>2009</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.0</td>
      <td>263</td>
      <td>38.486161</td>
      <td>-11.155298</td>
      <td>90</td>
      <td>63</td>
      <td>58</td>
      <td>1986</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>31.130847</td>
      <td>-1.825359</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 288 columns</p>
</div>




```python
y.head()
```




    0        functional
    1        functional
    2        functional
    3    non functional
    4        functional
    Name: status_group, dtype: object



## train test split


```python
from sklearn.model_selection import train_test_split

## Train test split
X_train, X_test, y_train,y_test  = train_test_split(X,y, random_state=6)
```

# MODEL

## Timer class to time model fitting


```python
## a timer to record how long a process takes
class Timer():
    ## def init
    def __init__(self,format_="%m/%d/%y - %I:%M:%S %p", 
                 start_timer=True,str_width=30,verbose=1):
        import tzlocal
        self._verbose = verbose
#         if verbose >1:
#             print('---'*20)
#             print('\tTIMER LOG')
#             print('---'*20)

        self.tz = tzlocal.get_localzone()
        self.fmt = format_
        self._str_width = str_width
        
        self.created_at = self.get_time()# get time'

        
        self._log = []
#         self.log(f"[i] Timer created at \t{self.created_at.strftime(self.fmt):>{self._str_width}}",
#                 print_=False)
       
        if start_timer:
            self.start()
        
        
    def log(self,msg='',print_= None):
        if print_ is None:
            print_ = self._verbose         
        if print_>0:
            print(msg)
        self._log.append(msg)
        
    def _fmt_time(self,time_to_fmt):
        time_str = f"{time_to_fmt.strftime(self.fmt):>{self._str_width}}"
        return time_str
        
    ## def get time method
    def get_time(self):
        import datetime as dt
        return dt.datetime.now(self.tz)

    ## def start
    def start(self):
        time = self.get_time()
        self.started_at = time
        msg = f"[i] Timer started at \t{self.started_at.strftime(self.fmt):>{self._str_width}}"
        self.log(msg,print_=None)#_log.append(msg)


        

    ## def stop
    def stop(self):
        time = self.get_time()
        self.ended_at = time
        self.duration = self.ended_at - self.started_at
        
    
        msg1 = f"[i] Timer ended at \t{self.ended_at.strftime(self.fmt):>{self._str_width}}"
        msg2 = f"\t\t\t- Total time = {self.duration}"

        if self._verbose==2:
            print_override=False
        else:
            print_override=True
                
        self.log(msg1,print_=print_override)#_log.append(msg1)

#         if self._verbose>1:
#             print('---'*20)
            

        self.log(msg2,print_=print_override)#,print_=True)
        if self._verbose==2:
            print(self.summary())
#
    def summary(self):
        dashes= '---'*20
#         print()
        
    
        summary = self._log.copy()
        
        summary =['\n',dashes,'\tTIMER LOG',dashes,
                  *summary,dashes]
        
#         summary.append()
#         summary.append(dashes)
        
#         if self._verbose>1:
#             summary.append(dashes)
        return '\n'.join(summary)
    
    
    def __repr__(self):
        return self.summary()
```


```python
timer = Timer(verbose=2)#str_width=40)

timer.start()
timer.stop()
# type(timer)
# timer.log('cheese',False)
# timer
```

    [i] Timer started at 	        05/10/20 - 03:45:43 PM
    [i] Timer started at 	        05/10/20 - 03:45:43 PM
    
    
    ------------------------------------------------------------
    	TIMER LOG
    ------------------------------------------------------------
    [i] Timer started at 	        05/10/20 - 03:45:43 PM
    [i] Timer started at 	        05/10/20 - 03:45:43 PM
    [i] Timer ended at 	        05/10/20 - 03:45:43 PM
    			- Total time = 0:00:00.000425
    ------------------------------------------------------------


## functions


```python

## Write a fucntion to evalute the model
import sklearn.metrics as metrics

def evaluate_model(y_true, y_pred,X_true,clf,importance_top_n):
    """Takes in your target, target predictions, classifier and an int for 
        number of features you want to display in a bar graph.
        Returns the bar graph of most important features 
        along with a confusion matrix from your classifier."""
    
    ## Classification Report / Scores 
    print(metrics.classification_report(y_true,y_pred))

    # subplots for confusion matrix and bar plot
    fig, ax = plt.subplots(figsize=(12,6),ncols=2)
    metrics.plot_confusion_matrix(clf,X_true,y_true,cmap="Blues",
                                  normalize='true', xticks_rotation='vertical', ax=ax[0])
    
    df_importance = pd.Series(clf.feature_importances_,index=X_train.columns)
    df_importance.sort_values(ascending=True).tail(importance_top_n).plot(
        kind='barh', ax=ax[1])
#     top_features = df_importance.sort_values(ascending=True).tail(importance_top_n)
    
    ax[0].set(title='Confusion Matrix')
    ax[1].set(title='Top Important Features')
    y_score = clf.predict_proba(X_true)[:,1]

    plt.tight_layout()
    plt.show()
    
#     return top_features
```


```python
## visualize the decision tree
def visualize_tree(clf,feature_names=None,class_names=['0','1','2'],
                   kws={},save_filename=None,format_='png',save_and_show=False):
    """Visualizes a sklearn tree using sklearn.tree.export_graphviz. Takes in fitted classifier,
        the feature names you want as a list (If none, then uses the column names from classifier),
        the class names for your target as a list, the filename to save it under, 
        the file format you want as a string (default is png), 
        and save and show as a boolean value. True to save file then display in notebook. 
        (default is to only save it.) 
        """
    from sklearn.tree import export_graphviz
    from IPython.display import SVG
    import graphviz #import Source
    from IPython.display import display
    
    if feature_names is None:
        feature_names=X_train.columns

    tree_viz_kws =  dict(out_file=None,rounded=True, rotate=False, filled = True)
    tree_viz_kws.update(kws)

    # tree.export_graphviz(dt) #if you wish to save the output to a dot file instead
    tree_data=export_graphviz(clf,feature_names=feature_names, 
                                   class_names=class_names,**tree_viz_kws)
    graph = graphviz.Source(tree_data,format=format_)#'png')
    
    if save_filename is not None:
        graph.render(save_filename)
        if save_and_show:
            display(graph)
        else:
            print(f'[i] Tree saved as {save_filename}.{format_}')
    else:
        display(graph)

#     display(SVG(graph.pipe(format=format_)))#'svg')))
```

## Vanilla Decisiontree


```python
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.preprocessing import OneHotEncoder
from IPython.display import Image  
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data


```


```python
clf = DecisionTreeClassifier(random_state=6) # no params decided yet until grid search

# fit data
clf.fit(X_train,y_train)
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=6, splitter='best')




```python
y_preds = clf.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_preds))
```

    Accuracy:  0.7533513926512468



```python
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
```

    1.0
    0.7533513926512468



```python
# need to do class imbalance
y_test.value_counts(normalize=True)
```




    functional                 0.552962
    non functional             0.381399
    functional needs repair    0.065639
    Name: status_group, dtype: float64




```python
evaluate_model(y_test,y_preds,X_test,clf,15)
```

                             precision    recall  f1-score   support
    
                 functional       0.81      0.79      0.80      7961
    functional needs repair       0.34      0.37      0.35       945
             non functional       0.76      0.76      0.76      5491
    
                   accuracy                           0.75     14397
                  macro avg       0.63      0.64      0.64     14397
               weighted avg       0.76      0.75      0.75     14397
    



![png](output_82_1.png)


### vanilla tree eval

- Class functional and nonfunction are scoring pretty well, 0.8 and 0.77. 
- Functional need repair not so much with 0.38 recall. - means it only correctly classed 35%, with 44% incorrectly classed as functional. Need higher recall to get some of thosed missed that need repair.
- Important features seem to makes sense, with lat/long and location being telling, along with quantity (whether the well is dry or not)
- Construction year is different because there are still a lot of 0s in that column
- Management group could be related to poor management quality and tactics
- Extraction type gravity and source type borehole could be related

- Most telling signs, location (long/lat), how much water is left in the well (quantity_grp), sealevel (gps height)
- Theres a lot of 0s in the construction year column - should i change the zeros to unknown and use cat, or fil with 1900 and use as numeric.


### this tree visual is hashed out to save on load time. saved png is in folder as "vanilla_tree.png"


```python
# timer = Timer(verbose=2)#str_width=40)
# timer.start()

# # visualize tree (save to file)
# # hashed out but can load with pickle below to save on load time


# vanilla_tree = visualize_tree(clf,save_filename="vanilla_tree")


# timer.stop()
```

## class imbalance SMOTE


```python
## Check class
y_train.value_counts(normalize=True)
```




    functional                 0.542428
    non functional             0.388437
    functional needs repair    0.069135
    Name: status_group, dtype: float64




```python
from imblearn.over_sampling import SMOTE
smote = SMOTE()
smote_X_train, smote_y_train = smote.fit_sample(X_train, y_train)
```

    //anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/externals/six.py:31: FutureWarning:
    
    The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).
    
    //anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/utils/deprecation.py:144: FutureWarning:
    
    The sklearn.neighbors.base module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.neighbors. Anything that cannot be imported from sklearn.neighbors is now part of the private API.
    
    //anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/utils/deprecation.py:87: FutureWarning:
    
    Function safe_indexing is deprecated; safe_indexing is deprecated in version 0.22 and will be removed in version 0.24.
    
    //anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/utils/deprecation.py:87: FutureWarning:
    
    Function safe_indexing is deprecated; safe_indexing is deprecated in version 0.22 and will be removed in version 0.24.
    



```python
pd.Series(smote_y_train).value_counts()
```




    non functional             23428
    functional needs repair    23428
    functional                 23428
    dtype: int64



##  Vanilla model with class imbalance resolved


```python
# instantiate
clf = DecisionTreeClassifier(random_state=6) # no params decided yet until grid search
```


```python
# fit data
clf.fit(smote_X_train,smote_y_train)
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=6, splitter='best')




```python
y_preds = clf.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_preds))
```

    Accuracy:  0.7496006112384525



```python
evaluate_model(y_test,y_preds,X_test,clf,15)
```

                             precision    recall  f1-score   support
    
                 functional       0.82      0.77      0.79      7961
    functional needs repair       0.31      0.43      0.36       945
             non functional       0.76      0.77      0.77      5491
    
                   accuracy                           0.75     14397
                  macro avg       0.63      0.66      0.64     14397
               weighted avg       0.76      0.75      0.75     14397
    



![png](output_95_1.png)



```python
# old vanilla scores
#                          precision    recall  f1-score   support

#              functional       0.80      0.80      0.80      7848
# functional needs repair       0.36      0.35      0.35      1018
#          non functional       0.76      0.77      0.76      5531

#                accuracy                           0.75     14397
#               macro avg       0.64      0.64      0.64     14397
#            weighted avg       0.75      0.75      0.75     14397
```

### comments on class imbalance vanilla tree

- The accuracy score is pretty much the same
- Functional precision went up, funtional repair went down
- Functional recall went down, functional repair recall went up
- Functional repair and nonfunction f1 score both went up


## Feature selection - drop least important cols after vanilla tree model

### cols not listed in top features


```python


cols_to_drop = ['region_code', 'district_code', 'lga', 'extraction_type',
       'extraction_type_class', 'payment',
       'payment_type', 'source', 'source_class']
                
df.drop(columns = cols_to_drop ,axis=1, inplace=True)
```


```python
df.columns
```




    Index(['amount_tsh', 'gps_height', 'longitude', 'latitude', 'basin', 'region',
           'population', 'construction_year', 'extraction_type_group',
           'management', 'management_group', 'water_quality', 'quality_group',
           'quantity', 'quantity_group', 'source_type', 'waterpoint_type',
           'waterpoint_type_group', 'status_group'],
          dtype='object')



cols to delete:
- basin - Geographic water basin ** - don't need, cat, using region code
- region - Geographic location ** - don't need, cat, using region code
- district_code - Geographic location (coded) ++ - don't need, using region code
- lga - Geographic location - local govt area, could be used for political question
- extraction_type - The kind of extraction the waterpoint uses ++ ** - don't need, has too many nondescript cats
- payment - What the water costs ** - don't need, repeat of payment type
- source - The source of the water ** - don't need, source type is more concise
- source_class - The source of the water ++ ** - don't need, source type has more descript groups

keep these for now
- extraction_type_group - The kind of extraction the waterpoint uses ++ ** - don't need, too many cats
- management - How the waterpoint is managed ++ ** - don't need, to many cats, have mgmt group to use instead
- water_quality - The quality of the water ** - don't need, less descript compared to quality group
- quantity - The quantity of water ** - don't need, repeat of quantity group
- waterpoint_type - The kind of waterpoint ** - don't need, waterpoint type group is more concise


## split df into X,y


```python
y = df['status_group']
X = df.drop('status_group', axis=1)
```

## one hot cats


```python
X = pd.get_dummies(X)
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount_tsh</th>
      <th>gps_height</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>population</th>
      <th>construction_year</th>
      <th>basin_Internal</th>
      <th>basin_Lake Nyasa</th>
      <th>basin_Lake Rukwa</th>
      <th>basin_Lake Tanganyika</th>
      <th>basin_Lake Victoria</th>
      <th>basin_Pangani</th>
      <th>basin_Rufiji</th>
      <th>basin_Ruvuma / Southern Coast</th>
      <th>basin_Wami / Ruvu</th>
      <th>region_Arusha</th>
      <th>region_Dar es Salaam</th>
      <th>region_Dodoma</th>
      <th>region_Iringa</th>
      <th>region_Kagera</th>
      <th>region_Kigoma</th>
      <th>region_Kilimanjaro</th>
      <th>region_Lindi</th>
      <th>region_Manyara</th>
      <th>region_Mara</th>
      <th>region_Mbeya</th>
      <th>region_Morogoro</th>
      <th>region_Mtwara</th>
      <th>region_Mwanza</th>
      <th>region_Pwani</th>
      <th>region_Rukwa</th>
      <th>region_Ruvuma</th>
      <th>region_Shinyanga</th>
      <th>region_Singida</th>
      <th>region_Tabora</th>
      <th>region_Tanga</th>
      <th>extraction_type_group_afridev</th>
      <th>extraction_type_group_gravity</th>
      <th>extraction_type_group_india mark ii</th>
      <th>extraction_type_group_india mark iii</th>
      <th>...</th>
      <th>water_quality_salty</th>
      <th>water_quality_salty abandoned</th>
      <th>water_quality_soft</th>
      <th>water_quality_unknown</th>
      <th>quality_group_colored</th>
      <th>quality_group_fluoride</th>
      <th>quality_group_good</th>
      <th>quality_group_milky</th>
      <th>quality_group_salty</th>
      <th>quality_group_unknown</th>
      <th>quantity_dry</th>
      <th>quantity_enough</th>
      <th>quantity_insufficient</th>
      <th>quantity_seasonal</th>
      <th>quantity_unknown</th>
      <th>quantity_group_dry</th>
      <th>quantity_group_enough</th>
      <th>quantity_group_insufficient</th>
      <th>quantity_group_seasonal</th>
      <th>quantity_group_unknown</th>
      <th>source_type_borehole</th>
      <th>source_type_dam</th>
      <th>source_type_other</th>
      <th>source_type_rainwater harvesting</th>
      <th>source_type_river/lake</th>
      <th>source_type_shallow well</th>
      <th>source_type_spring</th>
      <th>waterpoint_type_cattle trough</th>
      <th>waterpoint_type_communal standpipe</th>
      <th>waterpoint_type_communal standpipe multiple</th>
      <th>waterpoint_type_dam</th>
      <th>waterpoint_type_hand pump</th>
      <th>waterpoint_type_improved spring</th>
      <th>waterpoint_type_other</th>
      <th>waterpoint_type_group_cattle trough</th>
      <th>waterpoint_type_group_communal standpipe</th>
      <th>waterpoint_type_group_dam</th>
      <th>waterpoint_type_group_hand pump</th>
      <th>waterpoint_type_group_improved spring</th>
      <th>waterpoint_type_group_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>6000.0</td>
      <td>1390</td>
      <td>34.938093</td>
      <td>-9.856322</td>
      <td>109</td>
      <td>1999</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>1399</td>
      <td>34.698766</td>
      <td>-2.147466</td>
      <td>280</td>
      <td>2010</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>25.0</td>
      <td>686</td>
      <td>37.460664</td>
      <td>-3.821329</td>
      <td>250</td>
      <td>2009</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.0</td>
      <td>263</td>
      <td>38.486161</td>
      <td>-11.155298</td>
      <td>58</td>
      <td>1986</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>31.130847</td>
      <td>-1.825359</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 110 columns</p>
</div>



## train test split


```python
## Train test split
X_train, X_test, y_train,y_test  = train_test_split(X,y,random_state=6)
```

## class imbalance SMOTE


```python
## Check class
y_train.value_counts(normalize=True)
```




    functional                 0.542428
    non functional             0.388437
    functional needs repair    0.069135
    Name: status_group, dtype: float64




```python
smote = SMOTE()
smote_X_train, smote_y_train = smote.fit_sample(X_train, y_train)
```

    //anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/utils/deprecation.py:87: FutureWarning:
    
    Function safe_indexing is deprecated; safe_indexing is deprecated in version 0.22 and will be removed in version 0.24.
    
    //anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/utils/deprecation.py:87: FutureWarning:
    
    Function safe_indexing is deprecated; safe_indexing is deprecated in version 0.22 and will be removed in version 0.24.
    



```python
pd.Series(smote_y_train).value_counts()
```




    non functional             23428
    functional needs repair    23428
    functional                 23428
    dtype: int64



## grid search for best Decision tree params


```python
## Import GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
## Instantiate classifier
clf = DecisionTreeClassifier(random_state=6)
## Set up param grid
param_grid = {'criterion':['gini','entropy'],
             'max_depth':[None, 5, 3, 10],
             'max_features':['auto',3,5,10,20,50,80,110],
             'min_samples_leaf':[1,2,0.1,0.2,0.5]}

## Instantiate GridSearchCV
grid_clf = GridSearchCV(clf,param_grid)
```


```python
## hashed out to save on load time. took 17 minutes

# ## Start timer and fit search
# timer = Timer()
# timer.start()

# grid_clf.fit(smote_X_train, smote_y_train)
# ## Stop time and print best params
# timer.stop()
# grid_clf.best_params_
```

### pickle the tree grid to save load time


```python
# # pickle a file for quick save state load
# import pickle
# filename = 'gridsearch_tree_pickle'
# outfile = open(filename,'wb')
# pickle.dump(grid_clf,outfile)
# outfile.close()
```


```python
# # process to unpickle file
# infile = open(filename,'rb')
# new_grid_tree = pickle.load(infile)
# infile.close()
```


```python
# # test if pickle load worked 
# # new_vanilla_tree)
# print(new_grid_tree==grid_tree)
# print(new_grid_tree.best_params_)
```

## New Decisiontree with gridsearch params


```python
## Create a tree with the gridsearch's best params

best_params = {'criterion': 'entropy',
 'max_depth': None,
 'max_features': 110,
 'min_samples_leaf': 1}

grid_tree = DecisionTreeClassifier(**best_params, random_state=6)
grid_tree.fit(smote_X_train, smote_y_train)
print(grid_tree.score(X_test,y_test))

## Get predictions for train and test
y_preds = grid_tree.predict(X_test)

evaluate_model(y_test,y_preds,X_test,grid_tree,15)
```

    0.7432103910536917
                             precision    recall  f1-score   support
    
                 functional       0.81      0.77      0.79      7961
    functional needs repair       0.31      0.42      0.35       945
             non functional       0.76      0.75      0.76      5491
    
                   accuracy                           0.74     14397
                  macro avg       0.62      0.65      0.63     14397
               weighted avg       0.75      0.74      0.75     14397
    



![png](output_122_1.png)



```python
# old smote tree results
#                          precision    recall  f1-score   support

#              functional       0.81      0.79      0.80      7848
# functional needs repair       0.35      0.40      0.37      1018
#          non functional       0.76      0.77      0.77      5531

#                accuracy                           0.75     14397
#               macro avg       0.64      0.65      0.65     14397
#            weighted avg       0.76      0.75      0.76     14397
```

### eval new tree model

- Functional precision went down, functional repair went up, 
- Functional recall went down, functional repair went up, nonfunc went down
- F1, functional and repair went down, nonfunc went up
- Looks like accuracy went down

### pickle the tree grid model


```python
# # pickle a file for quick save state load
# import pickle
# gridsearchtree = 'gridsearch_tree_model_pickle'
# outfile = open(gridsearchtree,'wb')
# pickle.dump(grid_tree,outfile)
# outfile.close()
```


```python
# # process to unpickle file
# infile = open(gridsearchtree,'rb')
# new_gridsearch_tree_model = pickle.load(infile)
# infile.close()
```


```python
# # test if pickle load worked 

# print(new_gridsearch_tree_model.score(X_test,y_test))

# # Get predictions for train and test
# y_preds = new_gridsearch_tree_model.predict(X_test)

# evaluate_model(y_test,y_preds,X_test,new_gridsearch_tree_model,15)
```

## random forest


```python
## Start timer and fit search
timer = Timer()
timer.start()


## Import Random Forest
from sklearn.ensemble import RandomForestClassifier
## Fit Random Forest
rf = RandomForestClassifier(random_state=6)
rf.fit(smote_X_train, smote_y_train)

## Get predictions and evaluate model
y_preds = rf.predict(X_test)
evaluate_model(y_test,y_preds,X_test, rf,15)

# check scores
print(rf.score(smote_X_train,smote_y_train))
print(rf.score(X_test,y_test))

## Stop time and print best params
timer.stop()


```

    [i] Timer started at 	        05/10/20 - 03:50:12 PM
    [i] Timer started at 	        05/10/20 - 03:50:12 PM
                             precision    recall  f1-score   support
    
                 functional       0.82      0.84      0.83      7961
    functional needs repair       0.37      0.41      0.39       945
             non functional       0.82      0.78      0.80      5491
    
                   accuracy                           0.79     14397
                  macro avg       0.67      0.68      0.67     14397
               weighted avg       0.79      0.79      0.79     14397
    



![png](output_131_1.png)


    0.9999857720106995
    0.787872473431965
    [i] Timer ended at 	        05/10/20 - 03:51:03 PM
    			- Total time = 0:00:50.561585



```python
#                          precision    recall  f1-score   support

#              functional       0.80      0.78      0.79      7887
# functional needs repair       0.34      0.44      0.38       961
#          non functional       0.76      0.75      0.76      5549

#                accuracy                           0.74     14397
#               macro avg       0.63      0.66      0.64     14397
#            weighted avg       0.75      0.74      0.75     14397
```

### comments random forest

- Precision, func went up, func repair up, nonfunc up
- Recall func down, func repair down, nonfunc up
- F1 func up, repair func down, nonfunc up
- Accuracy score went up
- Best performance model so far

## Grid search for random forest best params


```python
## Instantiate classifier
rf_clf = RandomForestClassifier(random_state=6)
## Set up param grid
param_grid = {'criterion':['gini','entropy'],
             'max_depth':[None, 5, 10],
             'max_features':['auto',3,20,50,100],
             'min_samples_leaf':[1,0.1,0.5]}

## Instantiate GridSearchCV
grid_rf = GridSearchCV(rf_clf,param_grid,cv=3)
```


```python
# # hashed out to save on load time

# ## Start timer and fit search
# timer = Timer()
# timer.start()

# grid_rf.fit(smote_X_train, smote_y_train)
# ## Stop time and print best params
# timer.stop()
# grid_rf.best_params_
```

## create new random forest with gridsearch params


```python
## Create a random forest with the gridsearch's best params

best_params_ = {'criterion': 'entropy',
                 'max_depth': None,
                 'max_features': 20,
                 'min_samples_leaf': 1}

# hard code in best params 
grid_rf = RandomForestClassifier(**best_params_, random_state=6)
grid_rf.fit(smote_X_train, smote_y_train)

# check scores
print(grid_rf.score(smote_X_train,smote_y_train))
print(grid_rf.score(X_test,y_test))

y_preds = grid_rf.predict(X_test)
evaluate_model(y_test,y_preds,X_test, grid_rf,15)



```

    0.9999857720106995
    0.7880113912620685
                             precision    recall  f1-score   support
    
                 functional       0.82      0.84      0.83      7961
    functional needs repair       0.38      0.42      0.40       945
             non functional       0.82      0.78      0.80      5491
    
                   accuracy                           0.79     14397
                  macro avg       0.67      0.68      0.68     14397
               weighted avg       0.79      0.79      0.79     14397
    



![png](output_139_1.png)



```python
# old rf numbers
#                          precision    recall  f1-score   support

#              functional       0.82      0.83      0.83      7961
# functional needs repair       0.37      0.42      0.40       945
#          non functional       0.81      0.78      0.80      5491

#                accuracy                           0.79     14397
#               macro avg       0.67      0.68      0.67     14397
#            weighted avg       0.79      0.79      0.79     14397
```

### eval new randomforest grid model

- Judging from scores, looks like grid search rf is over fitting but I thought rf's were supposed to be resilient to that
- Recall, func went down
- F1 func repair went up, 
- Accuracy stayed the same
- So only slightly better. 

## XGBoost randomforest

### XGBRF with non SMOTE training data returns an error


```python
## Start timer and fit search
timer = Timer()
timer.start()

## import xgboost RF
from xgboost import XGBRFClassifier,XGBClassifier

## Fit and Evaluate
xgb_rf = XGBRFClassifier(random_state=6)
# xgb_rf.fit(smote_X_train, smote_y_train)
xgb_rf.fit(smote_X_train, smote_y_train)


# print(xgb_rf.score(smote_X_train,smote_y_train))
print(xgb_rf.score(X_train,y_train))

print(xgb_rf.score(X_test,y_test))

y_preds = xgb_rf.predict(X_test)

evaluate_model(y_test,y_preds,X_test,xgb_rf,15)

## Stop time
timer.stop()
```

    [i] Timer started at 	        05/10/20 - 03:52:32 PM
    [i] Timer started at 	        05/10/20 - 03:52:32 PM



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-73-71a6a2214391> in <module>
         13 
         14 # print(xgb_rf.score(smote_X_train,smote_y_train))
    ---> 15 print(xgb_rf.score(X_train,y_train))
         16 
         17 print(xgb_rf.score(X_test,y_test))


    //anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/base.py in score(self, X, y, sample_weight)
        367         """
        368         from .metrics import accuracy_score
    --> 369         return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        370 
        371 


    //anaconda3/envs/learn-env/lib/python3.6/site-packages/xgboost/sklearn.py in predict(self, data, output_margin, ntree_limit, validate_features)
        789                                                  output_margin=output_margin,
        790                                                  ntree_limit=ntree_limit,
    --> 791                                                  validate_features=validate_features)
        792         if output_margin:
        793             # If output_margin is active, simply return the scores


    //anaconda3/envs/learn-env/lib/python3.6/site-packages/xgboost/core.py in predict(self, data, output_margin, ntree_limit, pred_leaf, pred_contribs, approx_contribs, pred_interactions, validate_features)
       1282 
       1283         if validate_features:
    -> 1284             self._validate_features(data)
       1285 
       1286         length = c_bst_ulong()


    //anaconda3/envs/learn-env/lib/python3.6/site-packages/xgboost/core.py in _validate_features(self, data)
       1688 
       1689                 raise ValueError(msg.format(self.feature_names,
    -> 1690                                             data.feature_names))
       1691 
       1692     def get_split_value_histogram(self, feature, fmap='', bins=None, as_pandas=True):


    ValueError: feature_names mismatch: ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 'f58', 'f59', 'f60', 'f61', 'f62', 'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 'f70', 'f71', 'f72', 'f73', 'f74', 'f75', 'f76', 'f77', 'f78', 'f79', 'f80', 'f81', 'f82', 'f83', 'f84', 'f85', 'f86', 'f87', 'f88', 'f89', 'f90', 'f91', 'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98', 'f99', 'f100', 'f101', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107', 'f108', 'f109'] ['amount_tsh', 'gps_height', 'longitude', 'latitude', 'population', 'construction_year', 'basin_Internal', 'basin_Lake Nyasa', 'basin_Lake Rukwa', 'basin_Lake Tanganyika', 'basin_Lake Victoria', 'basin_Pangani', 'basin_Rufiji', 'basin_Ruvuma / Southern Coast', 'basin_Wami / Ruvu', 'region_Arusha', 'region_Dar es Salaam', 'region_Dodoma', 'region_Iringa', 'region_Kagera', 'region_Kigoma', 'region_Kilimanjaro', 'region_Lindi', 'region_Manyara', 'region_Mara', 'region_Mbeya', 'region_Morogoro', 'region_Mtwara', 'region_Mwanza', 'region_Pwani', 'region_Rukwa', 'region_Ruvuma', 'region_Shinyanga', 'region_Singida', 'region_Tabora', 'region_Tanga', 'extraction_type_group_afridev', 'extraction_type_group_gravity', 'extraction_type_group_india mark ii', 'extraction_type_group_india mark iii', 'extraction_type_group_mono', 'extraction_type_group_nira/tanira', 'extraction_type_group_other', 'extraction_type_group_other handpump', 'extraction_type_group_other motorpump', 'extraction_type_group_rope pump', 'extraction_type_group_submersible', 'extraction_type_group_swn 80', 'extraction_type_group_wind-powered', 'management_company', 'management_other', 'management_other - school', 'management_parastatal', 'management_private operator', 'management_trust', 'management_unknown', 'management_vwc', 'management_water authority', 'management_water board', 'management_wua', 'management_wug', 'management_group_commercial', 'management_group_other', 'management_group_parastatal', 'management_group_unknown', 'management_group_user-group', 'water_quality_coloured', 'water_quality_fluoride', 'water_quality_fluoride abandoned', 'water_quality_milky', 'water_quality_salty', 'water_quality_salty abandoned', 'water_quality_soft', 'water_quality_unknown', 'quality_group_colored', 'quality_group_fluoride', 'quality_group_good', 'quality_group_milky', 'quality_group_salty', 'quality_group_unknown', 'quantity_dry', 'quantity_enough', 'quantity_insufficient', 'quantity_seasonal', 'quantity_unknown', 'quantity_group_dry', 'quantity_group_enough', 'quantity_group_insufficient', 'quantity_group_seasonal', 'quantity_group_unknown', 'source_type_borehole', 'source_type_dam', 'source_type_other', 'source_type_rainwater harvesting', 'source_type_river/lake', 'source_type_shallow well', 'source_type_spring', 'waterpoint_type_cattle trough', 'waterpoint_type_communal standpipe', 'waterpoint_type_communal standpipe multiple', 'waterpoint_type_dam', 'waterpoint_type_hand pump', 'waterpoint_type_improved spring', 'waterpoint_type_other', 'waterpoint_type_group_cattle trough', 'waterpoint_type_group_communal standpipe', 'waterpoint_type_group_dam', 'waterpoint_type_group_hand pump', 'waterpoint_type_group_improved spring', 'waterpoint_type_group_other']
    expected f90, f35, f94, f12, f71, f61, f22, f81, f89, f31, f98, f28, f105, f91, f77, f17, f56, f42, f20, f6, f93, f57, f100, f92, f95, f55, f108, f97, f19, f70, f101, f58, f13, f21, f41, f46, f26, f106, f109, f53, f96, f62, f80, f59, f50, f33, f54, f107, f37, f36, f47, f9, f65, f27, f5, f29, f85, f1, f73, f11, f99, f102, f79, f88, f67, f38, f15, f103, f63, f7, f34, f39, f52, f16, f45, f3, f24, f78, f30, f48, f60, f10, f66, f43, f32, f51, f4, f14, f2, f64, f18, f8, f74, f69, f82, f83, f75, f25, f104, f76, f86, f68, f23, f72, f49, f0, f40, f84, f44, f87 in input data
    training data did not have the following fields: region_Tabora, extraction_type_group_india mark ii, region_Rukwa, extraction_type_group_other handpump, extraction_type_group_swn 80, management_other - school, region_Tanga, region_Ruvuma, waterpoint_type_hand pump, source_type_other, quantity_group_unknown, quantity_group_seasonal, management_vwc, region_Iringa, region_Manyara, source_type_spring, extraction_type_group_nira/tanira, waterpoint_type_improved spring, quantity_insufficient, waterpoint_type_communal standpipe, management_private operator, waterpoint_type_group_dam, source_type_dam, extraction_type_group_submersible, source_type_river/lake, basin_Lake Nyasa, quantity_group_insufficient, region_Dodoma, construction_year, management_water authority, basin_Lake Victoria, quantity_enough, basin_Internal, source_type_rainwater harvesting, management_unknown, extraction_type_group_other, extraction_type_group_other motorpump, extraction_type_group_rope pump, region_Kagera, region_Morogoro, waterpoint_type_group_cattle trough, waterpoint_type_communal standpipe multiple, region_Mtwara, waterpoint_type_group_other, management_other, management_water board, region_Kilimanjaro, management_group_user-group, region_Kigoma, quality_group_salty, water_quality_unknown, water_quality_fluoride, region_Mwanza, management_parastatal, water_quality_coloured, quality_group_unknown, region_Singida, water_quality_salty, region_Pwani, quantity_group_dry, basin_Lake Rukwa, region_Shinyanga, region_Dar es Salaam, management_wua, region_Mbeya, source_type_borehole, gps_height, waterpoint_type_group_hand pump, population, management_group_commercial, management_group_parastatal, quality_group_colored, quantity_unknown, water_quality_salty abandoned, region_Arusha, management_wug, water_quality_fluoride abandoned, quantity_dry, waterpoint_type_cattle trough, extraction_type_group_afridev, water_quality_milky, basin_Pangani, quality_group_fluoride, basin_Lake Tanganyika, management_trust, waterpoint_type_group_improved spring, waterpoint_type_dam, amount_tsh, basin_Rufiji, extraction_type_group_india mark iii, extraction_type_group_gravity, water_quality_soft, extraction_type_group_wind-powered, management_company, extraction_type_group_mono, region_Lindi, quantity_seasonal, longitude, quality_group_good, waterpoint_type_group_communal standpipe, region_Mara, quality_group_milky, management_group_unknown, latitude, basin_Wami / Ruvu, source_type_shallow well, basin_Ruvuma / Southern Coast, waterpoint_type_other, management_group_other, quantity_group_enough



```python
smote_X_train
```




    array([[0.00000000e+00, 1.20100000e+03, 3.68201503e+01, ...,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [2.50000000e+01, 4.70000000e+01, 3.90938017e+01, ...,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [6.00000000e+00, 1.53800000e+03, 3.74490016e+01, ...,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           ...,
           [3.00000000e+02, 1.20439619e+03, 3.76642084e+01, ...,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [0.00000000e+00, 4.30053584e+02, 3.90300429e+01, ...,
            0.00000000e+00, 0.00000000e+00, 5.26791798e-01],
           [0.00000000e+00, 0.00000000e+00, 3.35706116e+01, ...,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])



### turned SMOTE training data into dataframe. XGBRF was returning error if entered SMOTE training data as an array


```python
smote_X_train_df = pd.DataFrame(smote_X_train,columns=X_test.columns)

```


```python
smote_X_train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount_tsh</th>
      <th>gps_height</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>population</th>
      <th>construction_year</th>
      <th>basin_Internal</th>
      <th>basin_Lake Nyasa</th>
      <th>basin_Lake Rukwa</th>
      <th>basin_Lake Tanganyika</th>
      <th>basin_Lake Victoria</th>
      <th>basin_Pangani</th>
      <th>basin_Rufiji</th>
      <th>basin_Ruvuma / Southern Coast</th>
      <th>basin_Wami / Ruvu</th>
      <th>region_Arusha</th>
      <th>region_Dar es Salaam</th>
      <th>region_Dodoma</th>
      <th>region_Iringa</th>
      <th>region_Kagera</th>
      <th>region_Kigoma</th>
      <th>region_Kilimanjaro</th>
      <th>region_Lindi</th>
      <th>region_Manyara</th>
      <th>region_Mara</th>
      <th>region_Mbeya</th>
      <th>region_Morogoro</th>
      <th>region_Mtwara</th>
      <th>region_Mwanza</th>
      <th>region_Pwani</th>
      <th>region_Rukwa</th>
      <th>region_Ruvuma</th>
      <th>region_Shinyanga</th>
      <th>region_Singida</th>
      <th>region_Tabora</th>
      <th>region_Tanga</th>
      <th>extraction_type_group_afridev</th>
      <th>extraction_type_group_gravity</th>
      <th>extraction_type_group_india mark ii</th>
      <th>extraction_type_group_india mark iii</th>
      <th>...</th>
      <th>water_quality_salty</th>
      <th>water_quality_salty abandoned</th>
      <th>water_quality_soft</th>
      <th>water_quality_unknown</th>
      <th>quality_group_colored</th>
      <th>quality_group_fluoride</th>
      <th>quality_group_good</th>
      <th>quality_group_milky</th>
      <th>quality_group_salty</th>
      <th>quality_group_unknown</th>
      <th>quantity_dry</th>
      <th>quantity_enough</th>
      <th>quantity_insufficient</th>
      <th>quantity_seasonal</th>
      <th>quantity_unknown</th>
      <th>quantity_group_dry</th>
      <th>quantity_group_enough</th>
      <th>quantity_group_insufficient</th>
      <th>quantity_group_seasonal</th>
      <th>quantity_group_unknown</th>
      <th>source_type_borehole</th>
      <th>source_type_dam</th>
      <th>source_type_other</th>
      <th>source_type_rainwater harvesting</th>
      <th>source_type_river/lake</th>
      <th>source_type_shallow well</th>
      <th>source_type_spring</th>
      <th>waterpoint_type_cattle trough</th>
      <th>waterpoint_type_communal standpipe</th>
      <th>waterpoint_type_communal standpipe multiple</th>
      <th>waterpoint_type_dam</th>
      <th>waterpoint_type_hand pump</th>
      <th>waterpoint_type_improved spring</th>
      <th>waterpoint_type_other</th>
      <th>waterpoint_type_group_cattle trough</th>
      <th>waterpoint_type_group_communal standpipe</th>
      <th>waterpoint_type_group_dam</th>
      <th>waterpoint_type_group_hand pump</th>
      <th>waterpoint_type_group_improved spring</th>
      <th>waterpoint_type_group_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.0</td>
      <td>1201.0</td>
      <td>36.820150</td>
      <td>-3.369736</td>
      <td>120.0</td>
      <td>2011.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>25.0</td>
      <td>47.0</td>
      <td>39.093802</td>
      <td>-6.648229</td>
      <td>193.0</td>
      <td>2010.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>6.0</td>
      <td>1538.0</td>
      <td>37.449002</td>
      <td>-3.307395</td>
      <td>15.0</td>
      <td>2008.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.0</td>
      <td>1923.0</td>
      <td>35.357620</td>
      <td>-4.077079</td>
      <td>1.0</td>
      <td>2012.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.0</td>
      <td>1236.0</td>
      <td>37.103310</td>
      <td>-3.201124</td>
      <td>1.0</td>
      <td>2009.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 110 columns</p>
</div>



### XGBRF with SMOTE dataframe


```python
## Start timer and fit search
timer = Timer()
timer.start()

## import xgboost RF
from xgboost import XGBRFClassifier,XGBClassifier

## Fit and Evaluate
xgb_rf = XGBRFClassifier(random_state=6)
xgb_rf.fit(smote_X_train_df, smote_y_train)


print(xgb_rf.score(smote_X_train_df,smote_y_train))
print(xgb_rf.score(X_test,y_test))

y_preds = xgb_rf.predict(X_test)

evaluate_model(y_test,y_preds,X_test,xgb_rf,15)

## Stop time
timer.stop()
```

    [i] Timer started at 	        05/10/20 - 04:04:33 PM
    [i] Timer started at 	        05/10/20 - 04:04:33 PM
    0.6315093051050026
    0.6102660276446482
                             precision    recall  f1-score   support
    
                 functional       0.69      0.74      0.71      7961
    functional needs repair       0.15      0.48      0.23       945
             non functional       0.87      0.45      0.59      5491
    
                   accuracy                           0.61     14397
                  macro avg       0.57      0.56      0.51     14397
               weighted avg       0.72      0.61      0.63     14397
    



![png](output_151_1.png)


    [i] Timer ended at 	        05/10/20 - 04:07:15 PM
    			- Total time = 0:02:42.235801


### comments on XGB

> Looks like testing performed better than training. Don't know how.

## Grid search for XGB random forest


```python
## Instantiate classifier
xgb_rf = XGBRFClassifier(random_state=6)

## Set up param grid
param_grid = {
    "learning_rate": [0.1],
    'max_depth': [5],
    'min_child_weight': [10],
    'subsample': [ 0.7],
    'n_estimators': [5, 30, 100],
    'colsample_bynode': [0.8],
    'num_parallel_tree': [100],
    'objective': ['reg:squarederror']
}

## Instantiate GridSearchCV
grid_xgb_rf = GridSearchCV(xgb_rf,param_grid, scoring='accuracy', cv=None, n_jobs=1)
```


```python
# # hashed out to save on load time

# ## Start timer and fit search
# timer = Timer()
# timer.start()

# grid_xgb_rf.fit(smote_X_train, smote_y_train)
# ## Stop time and print best params
# timer.stop()
# grid_xgb_rf.best_params_
```

    [i] Timer started at 	        05/01/20 - 02:05:24 PM
    [i] Timer started at 	        05/01/20 - 02:05:24 PM
    [i] Timer ended at 	        05/01/20 - 02:26:00 PM
    			- Total time = 0:20:35.824650





    {'colsample_bynode': 0.8,
     'learning_rate': 0.1,
     'max_depth': 5,
     'min_child_weight': 10,
     'n_estimators': 30,
     'num_parallel_tree': 100,
     'objective': 'reg:squarederror',
     'subsample': 0.7}



## create new xgb random forest with gridsearch params 


```python
## Create an xgb random forest with the gridsearch's best params


best_params = {'colsample_bynode': 0.8,
             'learning_rate': 0.1,
             'max_depth': 5,
             'min_child_weight': 10,
             'n_estimators': 30,
             'num_parallel_tree': 100,
             'objective': 'reg:squarederror',
             'subsample': 0.7}

## Start timer and fit search
timer = Timer()
timer.start()

grid_xgbrf_bestparams = XGBRFClassifier(**best_params, random_state=6)
grid_xgbrf_bestparams.fit(smote_X_train_df, smote_y_train)

# check scores
print(grid_xgbrf_bestparams.score(smote_X_train_df,smote_y_train))
print(grid_xgbrf_bestparams.score(X_test,y_test))

y_preds = grid_xgbrf_bestparams.predict(X_test)
evaluate_model(y_test,y_preds,X_test, grid_xgbrf_bestparams,15)


## Stop time
timer.stop()
```

    [i] Timer started at 	        05/10/20 - 04:07:15 PM
    [i] Timer started at 	        05/10/20 - 04:07:15 PM
    0.6891184337829378
    0.6618740015280962
                             precision    recall  f1-score   support
    
                 functional       0.75      0.75      0.75      7961
    functional needs repair       0.17      0.48      0.26       945
             non functional       0.82      0.56      0.66      5491
    
                   accuracy                           0.66     14397
                  macro avg       0.58      0.60      0.56     14397
               weighted avg       0.74      0.66      0.69     14397
    



![png](output_158_1.png)


    [i] Timer ended at 	        05/10/20 - 04:08:43 PM
    			- Total time = 0:01:27.561826


### eval new randomforest grid model

- It's performing mediocre at best. Can barely predict above 50% for nonfunctional pumps.
- Random forest with best params peformed better with accuracy score of .79, compared to this .67.

# Final best model - Randomforest


```python
## Start timer and fit search
timer = Timer()
timer.start()


# ## Fit model
best_params = {'criterion': 'entropy',
                 'max_depth': None,
                 'max_features': 20,
                 'min_samples_leaf': 1}

# hard code in best params 
grid_rf = RandomForestClassifier(**best_params, random_state=6)
grid_rf.fit(smote_X_train, smote_y_train)

# eval model
y_preds = grid_rf.predict(X_test)
evaluate_model(y_test,y_preds,X_test, grid_rf,10)

## Stop time
timer.stop()
```

    [i] Timer started at 	        05/10/20 - 04:08:43 PM
    [i] Timer started at 	        05/10/20 - 04:08:43 PM
                             precision    recall  f1-score   support
    
                 functional       0.82      0.84      0.83      7961
    functional needs repair       0.38      0.42      0.40       945
             non functional       0.82      0.78      0.80      5491
    
                   accuracy                           0.79     14397
                  macro avg       0.67      0.68      0.68     14397
               weighted avg       0.79      0.79      0.79     14397
    



![png](output_162_1.png)


    [i] Timer ended at 	        05/10/20 - 04:10:03 PM
    			- Total time = 0:01:19.905944



```python
print(grid_rf.score(smote_X_train,smote_y_train))
print(grid_rf.score(X_test,y_test))
```

    0.9999857720106995
    0.7880113912620685


# iNTERPRET

### Partial Dependence Plots
Will show how each feature effects classification.


```python
top_features = pd.Series(grid_rf.feature_importances_,index=X_train.columns).sort_values(ascending=False).head(10)
feature_names = [i for i in X_test.columns if X_test[i].dtype in [np.int64] or [np.float64]]

```


```python
top_features
```




    longitude                        0.148220
    latitude                         0.144684
    gps_height                       0.062059
    construction_year                0.049454
    population                       0.043126
    quantity_dry                     0.041890
    quantity_group_dry               0.031925
    extraction_type_group_gravity    0.031025
    amount_tsh                       0.023807
    quantity_enough                  0.018462
    dtype: float64




```python
# don't know how to get indexes from top features so doing it manually
top_features_list = ['longitude','latitude', 'gps_height','construction_year', 'quantity_dry','population',
                     'quantity_group_dry','extraction_type_group_gravity','amount_tsh', 'quantity_group_enough']
```


```python
# install pdp if you don't have it
# !pip install pdpbox
```


```python
feature_names
```




    ['amount_tsh',
     'gps_height',
     'longitude',
     'latitude',
     'population',
     'construction_year',
     'basin_Internal',
     'basin_Lake Nyasa',
     'basin_Lake Rukwa',
     'basin_Lake Tanganyika',
     'basin_Lake Victoria',
     'basin_Pangani',
     'basin_Rufiji',
     'basin_Ruvuma / Southern Coast',
     'basin_Wami / Ruvu',
     'region_Arusha',
     'region_Dar es Salaam',
     'region_Dodoma',
     'region_Iringa',
     'region_Kagera',
     'region_Kigoma',
     'region_Kilimanjaro',
     'region_Lindi',
     'region_Manyara',
     'region_Mara',
     'region_Mbeya',
     'region_Morogoro',
     'region_Mtwara',
     'region_Mwanza',
     'region_Pwani',
     'region_Rukwa',
     'region_Ruvuma',
     'region_Shinyanga',
     'region_Singida',
     'region_Tabora',
     'region_Tanga',
     'extraction_type_group_afridev',
     'extraction_type_group_gravity',
     'extraction_type_group_india mark ii',
     'extraction_type_group_india mark iii',
     'extraction_type_group_mono',
     'extraction_type_group_nira/tanira',
     'extraction_type_group_other',
     'extraction_type_group_other handpump',
     'extraction_type_group_other motorpump',
     'extraction_type_group_rope pump',
     'extraction_type_group_submersible',
     'extraction_type_group_swn 80',
     'extraction_type_group_wind-powered',
     'management_company',
     'management_other',
     'management_other - school',
     'management_parastatal',
     'management_private operator',
     'management_trust',
     'management_unknown',
     'management_vwc',
     'management_water authority',
     'management_water board',
     'management_wua',
     'management_wug',
     'management_group_commercial',
     'management_group_other',
     'management_group_parastatal',
     'management_group_unknown',
     'management_group_user-group',
     'water_quality_coloured',
     'water_quality_fluoride',
     'water_quality_fluoride abandoned',
     'water_quality_milky',
     'water_quality_salty',
     'water_quality_salty abandoned',
     'water_quality_soft',
     'water_quality_unknown',
     'quality_group_colored',
     'quality_group_fluoride',
     'quality_group_good',
     'quality_group_milky',
     'quality_group_salty',
     'quality_group_unknown',
     'quantity_dry',
     'quantity_enough',
     'quantity_insufficient',
     'quantity_seasonal',
     'quantity_unknown',
     'quantity_group_dry',
     'quantity_group_enough',
     'quantity_group_insufficient',
     'quantity_group_seasonal',
     'quantity_group_unknown',
     'source_type_borehole',
     'source_type_dam',
     'source_type_other',
     'source_type_rainwater harvesting',
     'source_type_river/lake',
     'source_type_shallow well',
     'source_type_spring',
     'waterpoint_type_cattle trough',
     'waterpoint_type_communal standpipe',
     'waterpoint_type_communal standpipe multiple',
     'waterpoint_type_dam',
     'waterpoint_type_hand pump',
     'waterpoint_type_improved spring',
     'waterpoint_type_other',
     'waterpoint_type_group_cattle trough',
     'waterpoint_type_group_communal standpipe',
     'waterpoint_type_group_dam',
     'waterpoint_type_group_hand pump',
     'waterpoint_type_group_improved spring',
     'waterpoint_type_group_other']




```python
# from sklearn.inspection import plot_partial_dependence as pdp

from pdpbox import pdp, get_dataset, info_plots
    
for feat in top_features_list:
    
    pdp_dist = pdp.pdp_isolate(model=grid_rf, dataset=X_test, model_features=feature_names, feature=feat)
    
    plot_params = {
            # plot title and subtitle
            'title': 'PDP for feature "%s"' % feat,
#             'subtitle': "Number of unique grid points: %d" % n_grids,
            'title_fontsize': 15,
            'subtitle_fontsize': 12,
            'font_family': 'Arial',
            # matplotlib color map for ICE lines
            'line_cmap': 'Blues',
            'xticks_rotation': -90,
            # pdp line color, highlight color and line width
            'pdp_color': '#1A4E5D',
            'pdp_hl_color': '#FEDC00',
            'pdp_linewidth': 1.5,
            # horizon zero line color and with
            'zero_color': '#E75438',
            'zero_linewidth': 1,
            # pdp std fill color and alpha
            'fill_color': '#66C2D7',
            'fill_alpha': 0.2,
            # marker size for pdp line
            'markersize': 3.5}
    
    pdp.pdp_plot(pdp_dist, feat, figsize=(20,6), ncols=3, plot_params=plot_params)
    plt.show()
```


![png](output_171_0.png)



![png](output_171_1.png)



![png](output_171_2.png)



![png](output_171_3.png)



![png](output_171_4.png)



![png](output_171_5.png)



![png](output_171_6.png)



![png](output_171_7.png)



![png](output_171_8.png)



![png](output_171_9.png)



```python
# fig =  pdp.pdp_plot(pdp_dist, feat)
# fig[0].get_axes()
```




    [<matplotlib.axes._subplots.AxesSubplot at 0x1c1acf9208>,
     <matplotlib.axes._subplots.AxesSubplot at 0x1c1fe48d30>,
     <matplotlib.axes._subplots.AxesSubplot at 0x1c1fe48860>,
     <matplotlib.axes._subplots.AxesSubplot at 0x1c21507eb8>]




![png](output_172_1.png)


# CONCLUSIONS & RECOMMENDATIONS

Location (Latitude and Longitude): 
> Wells are classified as functioning between 34-37 degrees longitude. You'll see that backed up by viewing the classification of nonfunctional for longitude as being near opposite in results, showing most telling below 34 deg longitude and above 37 degrees. Latitude classifies functioning best at -9.5 and -3 degrees. Repairs are needed mainly at pumps along latitude -11.5 degrees. Nonfunctioning pumps will be along latitudes from -9 to -1. Nonfunctional pumps will be located mostly between 250 and 1250 sea level. Perhaps a well has to work harder and requires more materials if the pump is higher above sea level which means more things that could break. 

Population:
> It seems that the lower population and remote wells are more likely to be nonfunctional. Could be because of less people around to notify authorities of the problem and is not needed. Pumps near high population on the other hand, have a higher chance of needing repair. So it'd be best to focus on repairs to wells near populations who need it most and then get to non-populated pumps second.

Gravity extraction pumps:
> Gravity extraction type pumps have a high chance of needing repair, but are often still functional, so check on those pumps as priority and also focus on not building more of that type, focusing on more resilient type of pumps. 

For future feature selection and engineering: 
- Look at skewed 0s in long, lat that I removed, population, cnstruction year and amount tsh. 
- Construction year is very skewed but still important because the older wells could be falling apart. You'll see that the older a pump is the higher chance of it being nonfunctional or in need of repair. 
- If the well is dry, then the pump is most likely broken or in need of repair, although fixing it should be lower priority since it is not in use.  
- Lower total static head amount shows a higher chance of being broken. Pumps at 0 tsh are most likely broken. 
- Quantity labeled as enough most likely functioning. 


```python

```

# Competition run final test data


```python
testdata = pd.read_csv('testsetvalues.csv')
```


```python
df.columns
```




    Index(['amount_tsh', 'gps_height', 'longitude', 'latitude', 'basin', 'region',
           'population', 'construction_year', 'extraction_type_group',
           'management', 'management_group', 'water_quality', 'quality_group',
           'quantity', 'quantity_group', 'source_type', 'waterpoint_type',
           'waterpoint_type_group', 'status_group'],
          dtype='object')




```python
testdata.columns
```




    Index(['id', 'amount_tsh', 'date_recorded', 'funder', 'gps_height',
           'installer', 'longitude', 'latitude', 'wpt_name', 'num_private',
           'basin', 'subvillage', 'region', 'region_code', 'district_code', 'lga',
           'ward', 'population', 'public_meeting', 'recorded_by',
           'scheme_management', 'scheme_name', 'permit', 'construction_year',
           'extraction_type', 'extraction_type_group', 'extraction_type_class',
           'management', 'management_group', 'payment', 'payment_type',
           'water_quality', 'quality_group', 'quantity', 'quantity_group',
           'source', 'source_type', 'source_class', 'waterpoint_type',
           'waterpoint_type_group'],
          dtype='object')




```python
cols_to_drop = ['date_recorded', 'funder',
       'installer', 'wpt_name', 'num_private',
        'subvillage', 'region_code', 'district_code', 'lga',
       'ward', 'public_meeting', 'recorded_by',
       'scheme_management', 'scheme_name', 'permit',
       'extraction_type', 'extraction_type_class',
    'payment', 'payment_type',
       'source', 'source_class']
                
testdata.drop(columns = cols_to_drop ,axis=1, inplace=True)
```


```python
testdata.columns
```




    Index(['id', 'amount_tsh', 'gps_height', 'longitude', 'latitude', 'basin',
           'region', 'population', 'construction_year', 'extraction_type_group',
           'management', 'management_group', 'water_quality', 'quality_group',
           'quantity', 'quantity_group', 'source_type', 'waterpoint_type',
           'waterpoint_type_group'],
          dtype='object')



### drop 0 deg long and -2e-8 lat rows 


```python
for col in testdata.columns:
    print(testdata[col].value_counts())
```

    4094     1
    39654    1
    35588    1
    58105    1
    27384    1
            ..
    1514     1
    5608     1
    26086    1
    19939    1
    65536    1
    Name: id, Length: 14850, dtype: int64
    0.0         10410
    500.0         772
    50.0          631
    1000.0        370
    20.0          349
                ...  
    200000.0        1
    100000.0        1
    70000.0         1
    2550.0          1
    7000.0          1
    Name: amount_tsh, Length: 68, dtype: int64
     0       5211
    -19        18
     1294      18
     1343      18
     1283      17
             ... 
     722        1
     698        1
     2126       1
     674        1
     2023       1
    Name: gps_height, Length: 2157, dtype: int64
    0.000000     457
    37.302281      2
    32.920579      2
    37.260069      2
    39.080573      2
                ... 
    30.926134      1
    37.227967      1
    36.032819      1
    35.628949      1
    35.894087      1
    Name: longitude, Length: 14390, dtype: int64
    -2.000000e-08    457
    -7.105919e+00      2
    -2.474560e+00      2
    -7.170666e+00      2
    -6.990042e+00      2
                    ... 
    -9.320133e+00      1
    -9.114386e+00      1
    -3.134371e+00      1
    -3.885609e+00      1
    -8.477215e+00      1
    Name: latitude, Length: 14390, dtype: int64
    Lake Victoria              2623
    Pangani                    2203
    Rufiji                     2011
    Internal                   1857
    Lake Tanganyika            1620
    Wami / Ruvu                1590
    Lake Nyasa                 1247
    Ruvuma / Southern Coast    1094
    Lake Rukwa                  605
    Name: basin, dtype: int64
    Shinyanga        1311
    Iringa           1305
    Mbeya            1119
    Kilimanjaro      1115
    Morogoro         1032
    Kagera            858
    Mwanza            795
    Arusha            761
    Kigoma            717
    Pwani             696
    Ruvuma            666
    Tanga             639
    Dodoma            578
    Tabora            507
    Mara              482
    Singida           443
    Rukwa             434
    Mtwara            414
    Manyara           389
    Lindi             374
    Dar es Salaam     215
    Name: region, dtype: int64
    0       5453
    1       1757
    150      436
    200      430
    250      406
            ... 
    244        1
    252        1
    284        1
    2365       1
    7000       1
    Name: population, Length: 637, dtype: int64
    0       5260
    2010     669
    2009     663
    2008     630
    2000     487
    2006     421
    2007     373
    2011     335
    2004     294
    2003     293
    1995     269
    2002     268
    2005     264
    2012     263
    1999     243
    1985     232
    1978     230
    1998     224
    1990     222
    1996     209
    1994     202
    1980     194
    1984     191
    1972     184
    1982     182
    1997     177
    1992     167
    2001     140
    1974     138
    1993     137
    1988     136
    1975     124
    1986     119
    1976     111
    1983     106
    1991      83
    1970      82
    1989      80
    1987      68
    1981      53
    1979      53
    1977      45
    1973      43
    2013      33
    1971      32
    1963      22
    1960      22
    1969      18
    1967      18
    1968      16
    1964       8
    1961       7
    1962       6
    1966       2
    1965       2
    Name: construction_year, dtype: int64
    gravity            6483
    nira/tanira        2051
    other              1672
    submersible        1593
    swn 80              918
    mono                763
    india mark ii       629
    afridev             438
    rope pump           121
    other handpump       83
    india mark iii       37
    wind-powered         35
    other motorpump      27
    Name: extraction_type_group, dtype: int64
    vwc                 10117
    wug                  1593
    water board           755
    wua                   583
    private operator      533
    parastatal            461
    other                 239
    water authority       219
    company               174
    unknown               122
    trust                  27
    other - school         27
    Name: management, dtype: int64
    user-group    13048
    commercial      953
    parastatal      461
    other           266
    unknown         122
    Name: management_group, dtype: int64
    soft                  12687
    salty                  1226
    unknown                 469
    milky                   201
    coloured                133
    salty abandoned          84
    fluoride                 44
    fluoride abandoned        6
    Name: water_quality, dtype: int64
    good        12687
    salty        1310
    unknown       469
    milky         201
    colored       133
    fluoride       50
    Name: quality_group, dtype: int64
    enough          8336
    insufficient    3767
    dry             1536
    seasonal        1025
    unknown          186
    Name: quantity, dtype: int64
    enough          8336
    insufficient    3767
    dry             1536
    seasonal        1025
    unknown          186
    Name: quantity_group, dtype: int64
    shallow well            4316
    spring                  4195
    borehole                2981
    river/lake              2537
    rainwater harvesting     568
    dam                      184
    other                     69
    Name: source_type, dtype: int64
    communal standpipe             7106
    hand pump                      4396
    other                          1630
    communal standpipe multiple    1508
    improved spring                 175
    cattle trough                    34
    dam                               1
    Name: waterpoint_type, dtype: int64
    communal standpipe    8614
    hand pump             4396
    other                 1630
    improved spring        175
    cattle trough           34
    dam                      1
    Name: waterpoint_type_group, dtype: int64



```python
testdata.shape
```




    (14850, 19)




```python
prediction_id = testdata['id']
testdata = testdata.drop('id', axis=1)
```


```python
prediction_id.head()
```




    0    50785
    1    51630
    2    17168
    3    45559
    4    49871
    Name: id, dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 57588 entries, 0 to 59399
    Data columns (total 19 columns):
    amount_tsh               57588 non-null float64
    gps_height               57588 non-null int64
    longitude                57588 non-null float64
    latitude                 57588 non-null float64
    basin                    57588 non-null object
    region                   57588 non-null object
    population               57588 non-null int64
    construction_year        57588 non-null int64
    extraction_type_group    57588 non-null object
    management               57588 non-null object
    management_group         57588 non-null object
    water_quality            57588 non-null object
    quality_group            57588 non-null object
    quantity                 57588 non-null object
    quantity_group           57588 non-null object
    source_type              57588 non-null object
    waterpoint_type          57588 non-null object
    waterpoint_type_group    57588 non-null object
    status_group             57588 non-null object
    dtypes: float64(3), int64(3), object(13)
    memory usage: 8.8+ MB



```python
testdata.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14850 entries, 0 to 14849
    Data columns (total 18 columns):
    amount_tsh               14850 non-null float64
    gps_height               14850 non-null int64
    longitude                14850 non-null float64
    latitude                 14850 non-null float64
    basin                    14850 non-null object
    region                   14850 non-null object
    population               14850 non-null int64
    construction_year        14850 non-null int64
    extraction_type_group    14850 non-null object
    management               14850 non-null object
    management_group         14850 non-null object
    water_quality            14850 non-null object
    quality_group            14850 non-null object
    quantity                 14850 non-null object
    quantity_group           14850 non-null object
    source_type              14850 non-null object
    waterpoint_type          14850 non-null object
    waterpoint_type_group    14850 non-null object
    dtypes: float64(3), int64(3), object(12)
    memory usage: 2.0+ MB



```python
onehot_testdata = pd.get_dummies(testdata)
onehot_testdata.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount_tsh</th>
      <th>gps_height</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>population</th>
      <th>construction_year</th>
      <th>basin_Internal</th>
      <th>basin_Lake Nyasa</th>
      <th>basin_Lake Rukwa</th>
      <th>basin_Lake Tanganyika</th>
      <th>basin_Lake Victoria</th>
      <th>basin_Pangani</th>
      <th>basin_Rufiji</th>
      <th>basin_Ruvuma / Southern Coast</th>
      <th>basin_Wami / Ruvu</th>
      <th>region_Arusha</th>
      <th>region_Dar es Salaam</th>
      <th>region_Dodoma</th>
      <th>region_Iringa</th>
      <th>region_Kagera</th>
      <th>region_Kigoma</th>
      <th>region_Kilimanjaro</th>
      <th>region_Lindi</th>
      <th>region_Manyara</th>
      <th>region_Mara</th>
      <th>region_Mbeya</th>
      <th>region_Morogoro</th>
      <th>region_Mtwara</th>
      <th>region_Mwanza</th>
      <th>region_Pwani</th>
      <th>region_Rukwa</th>
      <th>region_Ruvuma</th>
      <th>region_Shinyanga</th>
      <th>region_Singida</th>
      <th>region_Tabora</th>
      <th>region_Tanga</th>
      <th>extraction_type_group_afridev</th>
      <th>extraction_type_group_gravity</th>
      <th>extraction_type_group_india mark ii</th>
      <th>extraction_type_group_india mark iii</th>
      <th>...</th>
      <th>water_quality_salty</th>
      <th>water_quality_salty abandoned</th>
      <th>water_quality_soft</th>
      <th>water_quality_unknown</th>
      <th>quality_group_colored</th>
      <th>quality_group_fluoride</th>
      <th>quality_group_good</th>
      <th>quality_group_milky</th>
      <th>quality_group_salty</th>
      <th>quality_group_unknown</th>
      <th>quantity_dry</th>
      <th>quantity_enough</th>
      <th>quantity_insufficient</th>
      <th>quantity_seasonal</th>
      <th>quantity_unknown</th>
      <th>quantity_group_dry</th>
      <th>quantity_group_enough</th>
      <th>quantity_group_insufficient</th>
      <th>quantity_group_seasonal</th>
      <th>quantity_group_unknown</th>
      <th>source_type_borehole</th>
      <th>source_type_dam</th>
      <th>source_type_other</th>
      <th>source_type_rainwater harvesting</th>
      <th>source_type_river/lake</th>
      <th>source_type_shallow well</th>
      <th>source_type_spring</th>
      <th>waterpoint_type_cattle trough</th>
      <th>waterpoint_type_communal standpipe</th>
      <th>waterpoint_type_communal standpipe multiple</th>
      <th>waterpoint_type_dam</th>
      <th>waterpoint_type_hand pump</th>
      <th>waterpoint_type_improved spring</th>
      <th>waterpoint_type_other</th>
      <th>waterpoint_type_group_cattle trough</th>
      <th>waterpoint_type_group_communal standpipe</th>
      <th>waterpoint_type_group_dam</th>
      <th>waterpoint_type_group_hand pump</th>
      <th>waterpoint_type_group_improved spring</th>
      <th>waterpoint_type_group_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.0</td>
      <td>1996</td>
      <td>35.290799</td>
      <td>-4.059696</td>
      <td>321</td>
      <td>2012</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>1569</td>
      <td>36.656709</td>
      <td>-3.309214</td>
      <td>300</td>
      <td>2000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.0</td>
      <td>1567</td>
      <td>34.767863</td>
      <td>-5.004344</td>
      <td>500</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.0</td>
      <td>267</td>
      <td>38.058046</td>
      <td>-9.418672</td>
      <td>250</td>
      <td>1987</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>500.0</td>
      <td>1260</td>
      <td>35.006123</td>
      <td>-10.950412</td>
      <td>60</td>
      <td>2000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 110 columns</p>
</div>



# final data predictions for contest


```python
y_preds = grid_rf.predict(onehot_testdata)
predictions = pd.DataFrame(y_preds)
predictions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>non functional</td>
    </tr>
    <tr>
      <td>1</td>
      <td>functional needs repair</td>
    </tr>
    <tr>
      <td>2</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>3</td>
      <td>non functional</td>
    </tr>
    <tr>
      <td>4</td>
      <td>functional</td>
    </tr>
  </tbody>
</table>
</div>




```python
predictions.columns =['status_group'] 
```


```python
predictions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>status_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>non functional</td>
    </tr>
    <tr>
      <td>1</td>
      <td>functional needs repair</td>
    </tr>
    <tr>
      <td>2</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>3</td>
      <td>non functional</td>
    </tr>
    <tr>
      <td>4</td>
      <td>functional</td>
    </tr>
  </tbody>
</table>
</div>




```python
prediction_id = pd.DataFrame(prediction_id)
```


```python
prediction_id.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>50785</td>
    </tr>
    <tr>
      <td>1</td>
      <td>51630</td>
    </tr>
    <tr>
      <td>2</td>
      <td>17168</td>
    </tr>
    <tr>
      <td>3</td>
      <td>45559</td>
    </tr>
    <tr>
      <td>4</td>
      <td>49871</td>
    </tr>
  </tbody>
</table>
</div>




```python
prediction_id['status_group'] = predictions['status_group']
prediction_id.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>status_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>50785</td>
      <td>non functional</td>
    </tr>
    <tr>
      <td>1</td>
      <td>51630</td>
      <td>functional needs repair</td>
    </tr>
    <tr>
      <td>2</td>
      <td>17168</td>
      <td>functional</td>
    </tr>
    <tr>
      <td>3</td>
      <td>45559</td>
      <td>non functional</td>
    </tr>
    <tr>
      <td>4</td>
      <td>49871</td>
      <td>functional</td>
    </tr>
  </tbody>
</table>
</div>




```python
prediction_id.shape
```




    (14850, 2)




```python
# prediction_id.to_csv ('predictiondf_2.csv', index = False, header=True)
```
