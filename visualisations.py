import numpy as np
import pandas as pd
from IdealMetadataInterface import IdealMetadataInterface
import matplotlib.pyplot as plt
#import scipy.stats as stats
import seaborn as sns
import statistics
from sklearn.cluster import KMeans  
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask, render_template
import datetime
import requests
#from matplotlib import rcParams
metadatadir='/Users/athmika/work/Dissertation/data/DS_10283_3647 (1)/metadata_and_surveys/metadata/'
mdi = IdealMetadataInterface(metadatadir)
features = pd.read_csv("features.csv")
print("--------")
print("FEATURES:")
print(features)
#fig, ax = plt.subplots(1,2,figsize=(13,5))
meter_df = mdi.metadata.meterreading["meterreading"]

def init(meter_df):
    print("init:")
    # initialize the metadata interface   
    meter_df = mdi.metadata.meterreading["meterreading"]                                                                                                                                      
    tariff_df = mdi.metadata.tariffs['tariff']
    #meter_df = mdi.metadata.meterreading['meterreading']
    meter_df = meter_df[meter_df.energytype == "electricity"]
    meter_df.sort_values(by=['homeid'])
    tariff_df2 = tariff_df[(tariff_df['energytype'] == "electricity")]
    tariff_df3 = tariff_df2.groupby('homeid')['unit_charge_pence_per_kwh'].mean()
    tariff_df3 = tariff_df3.reset_index()

    print("tariff_df3",tariff_df3)
    tariff_df3.unit_charge_pence_per_kwh = tariff_df3['unit_charge_pence_per_kwh'].fillna(tariff_df3['unit_charge_pence_per_kwh'].mean())
    plt.figure()
    plt.title("Tariff Distribution")
    plt.scatter(tariff_df3['homeid'],tariff_df3['unit_charge_pence_per_kwh'])
    plt.xlabel("Home ID")
    plt.ylabel("Unit Charge Pence per KWH")
    #plt.show()
    
    plt.savefig('./static/images/scatter_plot1.png')
    print("scatter plot saved inside static/images")
    
    #return render_template('index.html', name = 'scatter_plot1', url ='/static/images/scatter_plot1.png')

# household energy consumption based on income bands
def energy_consumption_analysis():
    home_df = mdi.metadata.homes['home']
    home_df['income_band']
    #print(home_df.income_band.unique())
    income_groups = home_df.groupby('income_band')
    global i1,i2,i3,i4,i5
    i1 = pd.concat([income_groups.get_group("£13,500 to £16,199") , income_groups.get_group("£10,800 to £13,499"), income_groups.get_group("less than £10,800")])
    #print(i1.income_band.unique())
    i2 = pd.concat([income_groups.get_group("£16,200 to £19,799") , income_groups.get_group("£19,800 to £23,399"), income_groups.get_group("£23,400 to £26,999")])
    #print(i2.income_band.unique())
    i3 = pd.concat([income_groups.get_group("£27,000 to £32,399"), income_groups.get_group("£32,400 to £37,799"), income_groups.get_group("£37,800 to £43,199"),income_groups.get_group("Missing") ])
    #print("I3: ",i3.income_band.unique())
    i4 = pd.concat([income_groups.get_group("£43,200 to £48,599"),income_groups.get_group("£48,600 to £53,999"),income_groups.get_group("£54,000 to £65,999")])
    #print("I4: ",i4.income_band.unique())
    i5 = pd.concat([income_groups.get_group("£66,000 to £77,999"),income_groups.get_group("£78,000 to £89,999"), income_groups.get_group("£90,000 or more")])
    #print("I5: ",i5.income_band.unique())
    count_automation_1 = []
    count_automation_2 = []
    tariff_df = mdi.metadata.tariffs['tariff']
    tariff_df2 = tariff_df[(tariff_df['energytype'] == "electricity")]
    tariff_df3 = tariff_df2.groupby('homeid')['unit_charge_pence_per_kwh'].mean()
    tariff_df3 = tariff_df3.reset_index()
    tariff_df3 = tariff_df3.loc[tariff_df3.unit_charge_pence_per_kwh != 0.00]
    tariff_df3.unit_charge_pence_per_kwh = tariff_df3['unit_charge_pence_per_kwh'].fillna(tariff_df3['unit_charge_pence_per_kwh'].mean())
    i1 = i1[['homeid','install_type','income_band','residents','smart_automation','smart_monitors']]
    unit_charge = []
    
    for id in i1.homeid:
        if tariff_df3.loc[tariff_df3.homeid == id].empty == False:
            unit_charge.append(tariff_df3.loc[tariff_df3.homeid == id]['unit_charge_pence_per_kwh'].values[0])
        else:
            unit_charge.append(np.nan)
    i1['unit_charge_pence_per_kwh'] = unit_charge
    i1['unit_charge_pence_per_kwh'] = i1['unit_charge_pence_per_kwh'].fillna(i1['unit_charge_pence_per_kwh'].mean())
    i1["income_band"] = 1
    count_automation_1.append(len(i1[i1.smart_automation == "Own and use"]))
    count_automation_2.append(len(i1[i1.smart_automation == "Don't own"]))
    i2 = i2[['homeid','install_type','income_band','residents','smart_automation','smart_monitors']]
    unit_charge = []
    for id in i2.homeid:
        if tariff_df3.loc[tariff_df3.homeid == id].empty == False:
            unit_charge.append(tariff_df3.loc[tariff_df3.homeid == id]['unit_charge_pence_per_kwh'].values[0])
        else:
            unit_charge.append(np.nan)
    i2['unit_charge_pence_per_kwh'] = unit_charge
    i2.unit_charge_pence_per_kwh = i2['unit_charge_pence_per_kwh'].fillna(i2['unit_charge_pence_per_kwh'].mean())
    count_automation_1.append(len(i2[i2.smart_automation == "Own and use"]))
    count_automation_2.append(len(i2[i2.smart_automation == "Don't own"]))
    i2["income_band"] = 2
    i3 = i3[['homeid','install_type','income_band','residents','smart_automation','smart_monitors']]
    unit_charge = []
    for id in i3.homeid:
        if tariff_df3.loc[tariff_df3.homeid == id].empty == False:
            unit_charge.append(tariff_df3.loc[tariff_df3.homeid == id]['unit_charge_pence_per_kwh'].values[0])
        else:
            unit_charge.append(np.nan)
    i3['unit_charge_pence_per_kwh'] = unit_charge
    i3['unit_charge_pence_per_kwh'] = i3['unit_charge_pence_per_kwh'].fillna(i3['unit_charge_pence_per_kwh'].mean())
    count_automation_1.append(len(i3[i3.smart_automation == "Own and use"]))
    count_automation_2.append(len(i3[i3.smart_automation == "Don't own"]))
    i3["income_band"] = 3
    i4 = i4[['homeid','install_type','income_band','residents','smart_automation','smart_monitors']]
    unit_charge = []
    for id in i4.homeid:
        if tariff_df3.loc[tariff_df3.homeid == id].empty == False:
            unit_charge.append(tariff_df3.loc[tariff_df3.homeid == id]['unit_charge_pence_per_kwh'].values[0])
        else:
            unit_charge.append(np.nan)
    i4['unit_charge_pence_per_kwh'] = unit_charge
    i4['unit_charge_pence_per_kwh'] = i4['unit_charge_pence_per_kwh'].fillna(i4['unit_charge_pence_per_kwh'].mean())
    count_automation_1.append(len(i4[i4.smart_automation == "Own and use"]))
    count_automation_2.append(len(i4[i4.smart_automation == "Don't own"]))
    i4["income_band"] = 4
    i5 = i5[['homeid','install_type','income_band','residents','smart_automation','smart_monitors']]
    unit_charge = []
    for id in i5.homeid:
        if tariff_df3.loc[tariff_df3.homeid == id].empty == False:
            unit_charge.append(tariff_df3.loc[tariff_df3.homeid == id]['unit_charge_pence_per_kwh'].values[0])
        else:
            unit_charge.append(np.nan)
    i5['unit_charge_pence_per_kwh'] = unit_charge
    count_automation_1.append(len(i5[i5.smart_automation == "Own and use"]))
    count_automation_2.append(len(i5[i5.smart_automation == "Don't own"]))
    i5['unit_charge_pence_per_kwh'] = i5['unit_charge_pence_per_kwh'].fillna(i5['unit_charge_pence_per_kwh'].mean())
    i5["income_band"] = 5

    X = ['Lower income','Lower middle','Middle','Upper Middle', 'High income']
    X_axis = np.arange(len(X))
    plt.figure()
    plt.bar(X_axis - 0.2, count_automation_1, 0.5, label = 'Own and Use')
    plt.bar(X_axis + 0.2, count_automation_2, 0.5, label = 'Don\'t own')
    plt.xticks(X_axis, X)
    plt.xlabel("Income Groups")
    plt.ylabel("Number of households")
    plt.title("Distribution of income groups that use smart automation")
    plt.savefig('./static/images/smart_automation.png')
    print("saved bar graph")
    
        

def meter_reading(meter_df):
    # Meter reading
    home_df = mdi.metadata.homes['home']
    meter_df = meter_df.loc[meter_df["energytype"] == "electricity"]
    meter_df.sort_values("homeid")
    groups = meter_df.groupby('homeid').size()
    readings = {}

    for id in meter_df.homeid:
        if id in groups and groups[id] > 1:
            reading_values = meter_df[meter_df['homeid'] == id].reading.values[0:2]
            dates = meter_df[meter_df['homeid'] == id].date.values[0:2]
            year1,month1,day1 = dates[0].split('-')
            date1 = datetime.datetime(int(year1),int(month1),int(day1))
            year2,month2,day2 = dates[1].split('-')
            date2 = datetime.datetime(int(year2),int(month2),int(day2))
            date = date1-date2
            days = date.days
            if date.days < 1:
                days = date.days*(-1)
            rdg = round(reading_values[1] - reading_values[0],2)
            if rdg < 1:
                rdg = rdg * (-1)
            readings[id] = [days, rdg]
    monthly_readings = {}
    for r in readings:
        if readings[r][0] != 0 and 0 < readings[r][1] and 4000 > readings[r][1]:
            monthly_rdg = round(readings[r][1]/readings[r][0]*30,2)
            if monthly_rdg < 800:
                monthly_readings[int(r)] = monthly_rdg
        else:
            monthly_readings[int(r)] = 0
    vals = list(monthly_readings.values())
    avg_monthly_reading = statistics.mean(vals)
    for r in monthly_readings:
        if monthly_readings[r] == 0:
            monthly_readings[r] = round(avg_monthly_reading,2)
    plt.figure()
    X = monthly_readings.keys()
    Y = monthly_readings.values()
    plt.bar(X,Y,width=3)
    plt.xlabel("Home id")
    plt.ylabel("Avg Monthly Consumption in kwh")
    plt.title("Average household monthly consumption of electricity")
    plt.savefig("./static/images/consumption_vals")
    consumption_i1, consumption_i2, consumption_i3, consumption_i4, consumption_i5, consumption_i6 = [[] for i in range(6)]
    for id in monthly_readings:
        if id in i1.homeid:
            consumption_i1.append(monthly_readings[id])
        elif id in i2.homeid:
            consumption_i2.append(monthly_readings[id])
        elif id in i3.homeid:
            consumption_i3.append(monthly_readings[id])
        elif id in i4.homeid:
            consumption_i4.append(monthly_readings[id])
        elif id in i5.homeid:
            consumption_i5.append(monthly_readings[id])
        else:
            #print("present in none of i")
            consumption_i3.append(monthly_readings[id])
    
    plt.figure()
    plt.boxplot([consumption_i1,consumption_i2,consumption_i3, consumption_i4, consumption_i5])
    plt.xlabel("Income Groups")
    plt.ylabel("Monthly Consumption in KWH")
    plt.title("Monthly consumption of different income groups")
    plt.savefig("./static/images/boxplot1.png")
    # Consumption by family size

    small_families = home_df[home_df.residents <= 2]
    medium_families = home_df[(home_df.residents == 3) | (home_df.residents == 4)]
    large_families = home_df[home_df.residents > 4]
    small_families = small_families[["homeid","residents","location","income_band","urban_rural_name","occupied_days","occupied_nights"]]
    medium_families = medium_families[["homeid","residents","location","income_band","urban_rural_name","occupied_days","occupied_nights"]]
    large_families = large_families[["homeid","residents","location","income_band","urban_rural_name","occupied_days","occupied_nights"]]
    #Consumption analysis - small families
    c = []
    for id in small_families.homeid:
        if id in monthly_readings:
            c.append(monthly_readings[id]) 
        else:
            c.append(np.nan)
    small_families["monthly_consumption"] = c

    small_families_df = small_families.dropna()
    small_families["monthly_consumption"] = small_families["monthly_consumption"].fillna(small_families["monthly_consumption"].mean())
    #Medium Families 
    c = []
    for id in medium_families.homeid:
        if id in monthly_readings:

            c.append(monthly_readings[id]) 
        else:
            c.append(np.nan) 
    medium_families["monthly_consumption"] = c
    medium_families_df = medium_families.dropna()
    medium_families["monthly_consumption"] = medium_families["monthly_consumption"].fillna(medium_families["monthly_consumption"].mean())
    c = []
    for id in large_families.homeid:
        if id in monthly_readings:
            c.append(monthly_readings[id]) 
        else:
            c.append(np.nan) 
    large_families["monthly_consumption"] = c
    large_families_df = large_families.dropna()
    large_families["monthly_consumption"] = large_families["monthly_consumption"].fillna(large_families["monthly_consumption"].mean())
    #consumption_df 
    global consumption_df
    consumption_df = pd.concat([i1,i2,i3,i4,i5])
#consumption_df
    c = []
    for id in consumption_df.homeid:

        if id in monthly_readings:
            c.append(monthly_readings[id]) 
        else:
            #if()
            if(not(i1[i1.homeid == id].empty)):
                c.append(sum(consumption_i1)/len(consumption_i1))
            elif(not(i2[i2.homeid == id].empty)):
                c.append(sum(consumption_i2)/len(consumption_i2))
            elif(not(i3[i3.homeid == id].empty)):
                c.append(sum(consumption_i3)/len(consumption_i3))
            elif(not(i4[i4.homeid == id].empty)):
                c.append(sum(consumption_i4)/len(consumption_i4))
            elif(not(i5[i5.homeid == id].empty)):
                c.append(sum(consumption_i5)/len(consumption_i5))
            else:
                c.append(np.nan)


        #c.append(np.nan)
    consumption_df["monthly_consumption"] = c
    #Y = consumption_df.monthly_consumption
    plt.figure()
    plt.boxplot([small_families_df.monthly_consumption,medium_families_df.monthly_consumption,large_families_df.monthly_consumption])
    plt.xticks([1,2,3],['Small families', 'Medium Families', 'Large families'])
    plt.ylabel("Monthly Consumption in KWH")
    plt.title("Consumption distribution by family size")
    plt.savefig("./static/images/boxplot2.png")
    labels = "Lower Income","Lower Middle Income","Middle Income","Upper Middle Income","High Income"
    sizes = [i1.size,i2.size,i3.size,i4.size,i5.size]
    plt.figure()
    plt.pie(sizes,labels=labels,autopct='%1.1f%%') 
    plt.title("Income distribution of the 255 families")
    plt.savefig("./static/images/piechart1.png")

def clustering(): 
    #global features
    #features = pd.concat([i1,i2,i3,i4,i5])
    features = consumption_df
    #consumption_df["monthly_consumption"] = consumption_df["monthly_consumption"].fillna(consumption_df["monthly_consumption"].mean())
    #i1[i1.homeid == 120].empty
    print("features: ",features)
    x = features[["income_band","monthly_consumption","residents"]].values
    #Using for loop for iterations from 1 to 10.  
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state= 42)  
    y_predict= kmeans.fit_predict(x) 
    features["clusters"] = y_predict
    plt.figure()
    plt.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster  
    plt.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster  
    plt.scatter(x[y_predict == 2, 0], x[y_predict == 2, 1], s = 100, c = 'red', label = 'Cluster 3') #for third cluster  
    plt.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') #for fourth cluster   
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')   
    plt.title('2-D Clusters of households')  
    plt.xlabel("income band")
    plt.ylabel("Consumption (KWH)")
    #plt.xlabel('Annual Income (k$)')  
    #plt.ylabel('Spending Score (1-100)')  
    plt.legend()  
    plt.savefig("./static/images/2dcluster.png") 
    plt.figure()
    kplot = plt.axes(projection='3d')
    c1 = features[features.clusters == 0]
    c2 = features[features.clusters == 1]
    c3 = features[features.clusters == 2]
    c4 = features[features.clusters == 3]
 
    #Cluster1
    kplot.scatter3D(c1.income_band, c1.residents, c1.monthly_consumption, c='red', label = 'Cluster 1')

    #Cluster2
    kplot.scatter3D(c2.income_band, c2.residents, c2.monthly_consumption, c='cyan', label = 'Cluster 2')

    #Cluster1
    kplot.scatter3D(c3.income_band, c3.residents, c3.monthly_consumption, c='green', label = 'Cluster 3')

    #Cluster1
    kplot.scatter3D(c4.income_band, c4.residents, c4.monthly_consumption, c='blue', label = 'Cluster 4')
    kplot.zaxis.labelpad=0
    kplot.set_title("3-D Cluster of Households")
    kplot.set_zlabel("monthly_consumption",rotation=90)
    kplot.set_xlabel("income band")
    kplot.set_ylabel("residents")
    kplot.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=7)
    #plt.tight_layout()
    kplot.figure.savefig("./static/images/clusters.png",dpi=300)
    
    print("saved clusters")
    #visulaizing the clusters  
    
def analyse_carbon_emissions(features):
    
    features = pd.read_csv("features.csv")
    features_less = features[features.carbon_intensity.notna()]
    features_less["carbon_emissions"] = features_less["monthly_consumption"] * features_less["carbon_intensity"]
    plt.figure()
    plt.title("Cost vs CO2 Emissions (Monthly)")
    plt.scatter(features_less.monthly_cost,features_less.carbon_emissions)
    plt.xlabel("Monthly cost (pounds)")
    plt.ylabel("Carbon emissions (nCO2)")
    plt.savefig("./static/images/cost_emissions.png")

    plt.figure()
    plt.scatter(features_less.monthly_consumption,features_less.carbon_emissions)
    plt.xlabel("Monthly consumption (KWH)")
    plt.ylabel("Carbon Emissions (nCO2)")
    plt.title("Consumption vs CO2 Emissions (Average Monthly)")
    plt.savefig("./static/images/consumption_emissions.png")
 



    # COST VS UNIT CHARGE PENCE PER KWH
    plt.figure()
    plt.scatter(features.monthly_cost,features.unit_charge_pence_per_kwh)
    plt.xlabel("Monthly cost (pounds)")
    plt.ylabel("Unit Charge pence per KWH")
    plt.title("Monthly Tariff vs Unit Charge pence per kwh")
    plt.savefig("./static/images/cost_unitcharge.png")

    plt.figure()
    plt.scatter(features.homeid,features.monthly_cost)
    plt.xlabel("Home ID")
    plt.ylabel("Monthly Cost")
    plt.title("Distribution of Monthly Cost")
    plt.savefig("./static/images/monthly_cost.png")

          






init(metadatadir)
print("distribution")
energy_consumption_analysis()
meter_reading(meter_df)
clustering()
analyse_carbon_emissions(features)




