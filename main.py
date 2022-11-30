#flask dependencies 
from flask import Flask, jsonify, render_template, request as req
#from flask_cors import CORS
import json 
import pandas as pd
#import methodology
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots 
import plotly.io as pio

#methodology dependencies
import requests 
import numpy 
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)
#CORS(app)



####### GRAPHS ########
def my_plot(dataframe,plot_var):
    data_plot = go.Scatter(x=dataframe.index,y=dataframe.plot_var, line=dict(color="#CE285E",width=2))
    layout=go.Layout(title=dict(text="Tomorrow signals",x=0.5), 
                     xaxis_title="Hour", yaxis_title="Signals")
    fig =go.Figure(data=data_plot, layout=layout)

    #"This is a Line Chart of Variable"+" "+str(plot_var)
    
    # This is conversion step...
    fig_json = fig.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def my_plot_full_bar(dataframe):
    data_plot = px.bar(dataframe,x=dataframe.index,y=dataframe.columns, 
                       labels={'PBF_datetime':'Hour', 
                               'value':'MWh', 
                               'PBF_shortname':'Sources'},
                       title="<b>Day-ahead scheduled generation</b><br><i>Breakdown shows the energy scheduled by production type for the Spanish peninsular electrical system</i>")
    fig_json = data_plot.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def my_plot_full_bar_CO2(dataframe):
    data_plot = px.bar(dataframe,x=dataframe.index,y=dataframe.columns, 
                       labels={'PBF_datetime':'Hour', 
                               'value':'kgCO2', 
                               'PBF_shortname':'Sources (Click to hide)'},
                       title="<b>Day-ahead CO2 Emissions [kgCO2 eq.]<br>per technology for day-ahead scheduled generation</b>")
    data_plot.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-1.02,
    xanchor="center",
    x=0.5,
    entrywidth=20,
    font=dict(
            family="Arial",
            size=10,
            color="black"
        ),
    title=None,

))
    fig_json = data_plot.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def my_plot_full_bar_PE(dataframe):
    data_plot = px.bar(dataframe,x=dataframe.index,y=dataframe.columns, 
                       labels={'PBF_datetime':'Hour', 
                               'value':'kWh_pe', 
                               'PBF_shortname':'Sources (Click to hide)'},
                       title="<b>Day-ahead Primary energy use [kWh_pe eq.] <br>per technology for day-ahead scheduled generation </b>")
    data_plot.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-1.02,
    xanchor="center",
    x=0.5,
    entrywidth=20,
    font=dict(
            family="Arial",
            size=10,
            color="black"
        ),
    title=None,

))
    fig_json = data_plot.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def F(dataframe):
    data_plot = px.line(dataframe,x=dataframe.index,y=dataframe.columns, 
                        title='Modulation signals',
                        labels={'PBF_datetime':'Hour','value':'Modulation signals',
                                'PBF_shortname':'Signals'})
    fig_json = data_plot.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def pie_plot(dataframe):
    data_plot = px.pie(dataframe, values=dataframe['Total'], names="Renewables",
             color_discrete_sequence=px.colors.sequential.RdBu,
             opacity=0.7, hole=0.5)
    fig_json = data_plot.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def pie_subplots(dataframe):
    res = dataframe['RES-E-RATIO'].mean()
    nonres = 1 - res

    wind = dataframe['Wind'].mean()
    hydro = dataframe['Hydro'].mean()
    PV = dataframe['Photovoltaic'].mean()
    thermal = dataframe['Solar thermal'].mean()
    biogas = dataframe['Biogas'].mean()
    biomass = dataframe['Biomass'].mean()

    res_df_pie = pd.DataFrame({'Type': ['Renewables', 'Non Renewables'],
                            'Percentage': [res, nonres]})

    res_pie = pd.DataFrame({'Type': ['Wind', 'Hydro', 'Photovoltaic', 'Solar thermal', 'Biogas', 'Biomass'],
                            'Percentage': [wind,hydro,PV,thermal,biogas,biomass]})

    #create subplots
    fig = make_subplots(rows=1,cols=2,specs=[[{'type':'domain'},{'type':'domain'}]])

    #creating our pie charts
    fig.add_trace(go.Pie(labels=res_df_pie['Type'], values=res_df_pie['Percentage'], name=''),1,1)
    fig.add_trace(go.Pie(labels=res_pie['Type'], values=res_pie['Percentage'], name='RES SHARE'),1,2)

    #use hole a donut like
    fig.update_traces(hole=.6, hoverinfo="label+percent+name")

    fig.update_layout(
            title_text="<b>Day-ahead generation by renewable sources</b><br><i>Click to hide source</i>",
            #add annotations in the center of the donnut
            annotations=[dict(text='', x=0.18, y=0.5, font_size=20, showarrow=False),
                        dict(text='', x=0.80, y=0.5, font_size=20, showarrow=False)])

    fig_json = fig.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def dropdown_menu_line(dataframe):    
    x = dataframe.index  
    y1 = dataframe['AEF'].values.tolist()
    y2 = dataframe['APEF'].values.tolist()
    y3 = dataframe['MEFmodel'].values.tolist()
    y4 = dataframe['MPEFmodel'].values.tolist()

    
    plot = go.Figure(data=[
                        go.Scatter(name='Average Emissions Factor\n (AEF) [kgCO2/kWh]',x=x,y=y1),
                        go.Scatter(name='Average Primary Energy Factor (APEF) [kWpe/kWh]',x=x,y=y2),
                        go.Scatter(name='Marginal Emissions Factor (MEF) [kgCO2/kWh]',x=x,y=y3),
                        go.Scatter(name='Marginal Primary Energy Factor (MPEF) [kWpe/kWh]',x=x,y=y4)])

    # Add dropdown
    plot.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=list([
                    dict(label="Choose",
                        method="update",
                        args=[{"visible": [True, True, True, True]},
                            {"title": "Modulation signals (Click to hide)"}]),
                    dict(label="AEF",
                        method="update",
                        args=[{"visible": [True, False, False, False]},
                            {"title": "Average Emissions Factor [kgCO2/kWh]",
                                }]),
                    dict(label="APEF",
                        method="update",
                        args=[{"visible": [False, True, False, False]},
                            {"title": "Average Primary Energy Factor [kWpe/kWh]",
                                }]),
                    dict(label="MEF",
                        method="update",
                        args=[{"visible": [False, False, True, False]},
                            {"title": "Marginal Emissions Factor [kWpe/kWh]",
                                }]),
                    dict(label="MPEF",
                        method="update",
                        args=[{"visible": [False, False,False,True]},
                            {"title": "Marginal Primary Energy Factor [kWpe/kWh]",
                                }]),
                    dict(label="Emissions",
                        method="update",
                        args=[{"visible": [True, False,True,False]},
                            {"title": "Emissions (Average vs. Marginal)",
                                }]),
                    dict(label="Primary Energy",
                        method="update",
                        args=[{"visible": [False, True,False,True]},
                            {"title": "Primary Energy (Average vs. Marginal)",
                                }]),
                ]),
            )
        ])

    fig_json = plot.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def two_y_axis_dropdown(df):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    #plot set theme
    pio.templates.default = "plotly_white"


    # Add traces
    fig.add_trace(go.Scatter(x=df.index, y=df.AEF, name="AEF",mode='lines',line_color="blue"),secondary_y=False,)
    fig.add_trace(go.Scatter(x=df.index, y=df.APEF, name="APEF",line_color="black"),secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df.MEFmodel, name="MEF model",mode='lines+markers', line_color="blue"),secondary_y=False,)
    fig.add_trace(go.Scatter(x=df.index, y=df.MPEFmodel, name="MPEF model",mode='lines+markers',line_color="black"),secondary_y=True)


    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.6,
                y=1,
                active=0,
                buttons=list([
                    dict(label="All",
                        method="update",
                        args=[{"visible": [True, True, True, True]},
                            {"title": "<b>Day-ahead Modulation Signals</b> <br><i>Signals as inputs to activate energy flexiblity in buildings equipped with electric loads</i>",
                                }]),
                    dict(label="Emissions",
                        method="update",
                        args=[{"visible": [True, False, True, False]},
                            {"title": "Average vs. Marginal",
                                }]),
                    dict(label="Primary Energy",
                        method="update",
                        args=[{"visible": [False, True, False, True]},
                            {"title": "Average vs. Marginal",
                                }]),
                ]),
            )
        ], 
        legend=dict(orientation="h", y=-0.2, yanchor="bottom", xanchor="center",x=0.5))

    # Add figure title
    fig.update_layout(
        title_text="<b>Day-ahead Modulation Signals</b> <br><i>Signals as inputs to activate energy flexiblity in buildings equipped with electric loads</i>",
        #legend_title="Signals <br>(Click to hide)"
    )

    # Set x-axis title
    fig.update_xaxes(title_text=None)

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>kgCO2/kWh</b>", color="blue",secondary_y=False)
    fig.update_yaxes(title_text="<b>kWpe/kWh</b>", color="black", secondary_y=True)

    fig_json = fig.to_json()
    graphJSON = json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

####### FUNCTIONS ########
def get_values(df):
    #make API call
    TOKEN = open('Token.txt', 'r').read()
    headers = {'content-Type': 'application/json', 'Authorization': 'Token token={}'.format(TOKEN)}
    #Miurl = "https://api.esios.ree.es/indicators/3"
    #print(Miurl)
    
    market_info = [3,4,9,14,15,18,21,22,10064,10073,10086,10095,10167,10258,28,29,25,26,10104,10113,10122,10186,10141]
    #31 Import andorra 
    #30,16,10196
    print("Total of indicators: ", len(market_info))
    urls = []
    data = []
    
    for ids in market_info:
        Miurl = "https://api.esios.ree.es/indicators/"+str(ids)
        urls.append(Miurl)
        #print(Miurl)
        response = requests.get(Miurl, headers=headers).json()
        indicators = response['indicator']['short_name']
        print(indicators)
        #time.sleep(1)
        #print(response)
        
        for value in response['indicator']['values']:
            PBF_value = value['value']
            PBF_datetime = value['datetime']
            #PBF_datetime_utc = value['datetime_utc']
            #PBF_tz_time = value['tz_time']
            PBF_datetime = PBF_datetime.replace('Z', "")
            PBF_shortname = response['indicator']['short_name']
            #PBF_hour = PBF_datetime.replace('2022-05-14T', "")
            #PBF_hour = PBF_hour.replace(':00:00', "")

            #saving to pandas dataframe
            df = df.append({'PBF_shortname':PBF_shortname,
                            'PBF_value':PBF_value, 
                            'PBF_datetime':PBF_datetime},ignore_index=True)
                            #'PBF_datetime_utc':PBF_datetime_utc,
                            #'PBF_tz_time':PBF_tz_time}, ignore_index=True)

        
    return df 
       
def set_datetime_index(df):
    df.PBF_datetime = pd.to_datetime(df.PBF_datetime, utc=True)
    df = df.set_index(df.PBF_datetime)
    df = df.drop('PBF_datetime', axis=1)
    print('Datetimed')
    return df 

def pivot_sort_and_fill(df):
    #short df per date
    df = pd.pivot_table(df, values='PBF_value', index=['PBF_datetime'],columns=['PBF_shortname'])
    df = df.sort_values(by=['PBF_datetime']).fillna(0)
    print('Sorted')
    return df

def rename_en(df):
    df = df.rename(columns={'Biogás':'Biogas',
                        'Biomasa':'Biomass',
                        'Carbón':'Coal',
                        'Ciclo combinado':'Combined Gas Cycle',
                        'Cogeneración':'Natural Gas Cogeneration',
                        'Consumo bombeo':'Pump Consumption',
                        'Demanda Peninsular':'Peninsular demand',
                        'Derivados del petróleo ó carbón':'Fuel',
                        'Enlace Baleares':'Balearic link',
                        'Eólica':'Wind',
                        'Generación PBF total':'Total generation',
                        'Hidráulica':'Hydro',
                        'Importación Francia':'Import France',
                        'Importación Marruecos':'Import Morocco',
                        'Importación Portugal':'Import Portugal',
                        'Nuclear':'Nuclear',
                        'Océano y geotérmica':'Ocean/Geothermal',
                        'Residuos':'Waste RSU',
                        'Saldo Marruecos':'Morocco Exchange',
                        'Saldo Portugal':'Portugal Exchange',
                        'Saldo Francia':'France Exchange',
                        'Saldo interconexiones':'Interconnections exchange',
                        'Solar fotovoltaica':'Photovoltaic',
                        'Solar térmica':'Solar thermal',
                        'Turbinación bombeo':'Pumped hydro'
                       })
    print('Renamed')
    return df 

def build_df(df):
    column_list = ['Total generation', 
                   'Balearic link',
                    'Peninsular demand', 
                    'Total generation', 
                    'Nuclear', 
                    'Pumped hydro',
                    'Combined Gas Cycle', 
                    'Photovoltaic', 
                    'Solar thermal',
                    'Ocean/Geothermal', 
                    'Fuel', 
                    'Biomass', 
                    'Biogas', 
                    'Hydro', 
                    'Wind',
                    'Natural Gas Cogeneration', 
                    'Waste RSU', 
                    'Coal',
                    'Import France',
                    'Import Portugal', 
                    'Pump Consumption',
                    'Balearic link',
                    'Morocco Exchange',
                    'Portugal Exchange',
                    'France Exchange',
                    'Interconnections exchange']


    tech = []
    for col in column_list:
        if col not in df.columns:
            #tech = col
            #tech_no_available = tech.append()
            print(col)
            df[col] = numpy.nan
            
    df = df.fillna(0)  
    return df  

def renewables(df):
     #map renewables
    renewables = {'Wind', 
                  'Photovoltaic', 
                  'Solar thermal', 
                  'Biomass',
                  'Biogas', 
                  'Waste RSU', 
                  'Hydro', 
                  'Pumped hydro',
                  'Ocean/Geothermal', 
                }
    nonrenewables = {'Nuclear', 
                    'Coal', 
                    'Natural Gas Cogeneration', 
                    'Fuel', 
                    'Combined Gas Cycle'
                    }
    
    #RES-E-RATIO
    df['Renewables'] = ""
    df['NonRenewables'] = ""
    df['Renewables'] = df[(renewables)].sum(axis=1)
    df['NonRenewables'] = df[(nonrenewables)].sum(axis=1)
    df['Total'] = df[['Renewables', 'NonRenewables']].sum(axis=1)
    df['RES-E-RATIO'] = df['Renewables']/df['Total']
    
    return df    
    
def average_carbon_emissions(df):
    EF = {'ef_nuclear': 0.012, 'ef_coal': 1.210, 'ef_combined gas cycle': 0.492, 'ef_cogeneration': 0.380, 'ef_fuel': 0.866,
        'ef_wind': 0.014, 'ef_photovoltaic': 0.071, 'ef_solar thermal': 0.027, 'ef_ocean/geothermal': 0.082, 'ef_biomass':0.054,
        'ef_biogas':0.018,'ef_waste': 0.240, 'ef_hydro':0.024, 'ef_pumped hydro':0.062, 
        'ef_france': 0.068, 'ef_portugal': 0.484, 'ef_morocco': 0.729}

    #final energy values
    mapping_cfs = {'CO2_nuclear': ('Nuclear', 'ef_nuclear'), 
                'CO2_coal': ('Coal', 'ef_coal'), 
                'CO2_combinedgas': ('Combined Gas Cycle', 'ef_combined gas cycle'), 
                'CO2_cogeneration': ('Natural Gas Cogeneration', 'ef_cogeneration'), 
                'CO2_fuel': ('Fuel', 'ef_fuel'), 
                'CO2_wind': ('Wind', 'ef_wind'), 
                'CO2_photovoltaic': ('Photovoltaic', 'ef_photovoltaic'), 
                'CO2_solarthermal': ('Solar thermal', 'ef_solar thermal'),
                'CO2_oceangeothermal': ('Ocean/Geothermal', 'ef_ocean/geothermal'), 
                'CO2_biomass': ('Biomass', 'ef_biomass'),
                'CO2_biogas': ('Biogas', 'ef_biogas'), 
                'CO2_waste': ('Waste RSU', 'ef_waste'), 
                'CO2_hydroUGH': ('Hydro', 'ef_hydro'),
                'CO2_pumpedhydro': ('Pumped hydro', 'ef_pumped hydro'),
                'CO2_france': ('Import France', 'ef_france'), 
                'CO2_portugal': ('Import Portugal', 'ef_portugal')}

    co2_df = df.copy(deep=False)
    for column, data in mapping_cfs.items():
        co2_df[column] = co2_df[data[0]] * EF[data[1]]

    co2_df['Total Emisions'] = co2_df.drop(['Total generation', 
                                            'Balearic link',
                                            'Interconnections exchange',
                                            'Peninsular demand', 
                                            'Total generation', 
                                            'Nuclear', 
                                            'Pumped hydro',
                                            'Combined Gas Cycle', 
                                            'Photovoltaic', 
                                            'Solar thermal',
                                            'Ocean/Geothermal', 
                                            'Fuel', 
                                            'Biomass', 
                                            'Biogas', 
                                            'Hydro', 
                                            'Wind',
                                            'Natural Gas Cogeneration', 
                                            'Waste RSU', 
                                            'Coal',
                                            'Import France',
                                            'Import Portugal', 
                                            'Renewables',
                                            'NonRenewables', 
                                            'Total', 
                                            'RES-E-RATIO',
                                            'Pump Consumption',
                                            'Balearic link',
                                            'Morocco Exchange',
                                            'Portugal Exchange',
                                            'France Exchange',
                                            'Interconnections exchange'], 
                                           axis=1).sum(axis=1)

    co2_df['AEF'] = co2_df['Total Emisions']/co2_df['Peninsular demand']
    #df['AEF'] = co2_df['AEF']
    #df['Total Emisions'] = co2_df['Total Emisions']
    
    return co2_df 

def average_primary_energy(df):
    PEF = {'pef_nuclear': 3.030, 'pef_coal': 2.790, 'pef_combined gas cycle': 1.970, 'pef_cogeneration': 1.860, 
    'pef_fuel': 2.540, 'pef_wind': 0.030, 'pef_photovoltaic': 0.250, 'pef_solar thermal': 0.030,
    'pef_ocean/geothermal': 0.078, 'pef_biomass': 1.473,'pef_biogas': 2.790,'pef_waste': 1.473, 
    'pef_hydro': 0.100, 'pef_pumped hydro': 1.690, 'pef_france':2.553, 
    'pef_portugal': 1.587, 'pef_morocco': 2.200, 'pef_link': 0.340}

    #primary energy values
    mapping_cfs = {'PE_nuclear': ('Nuclear', 'pef_nuclear'), 
                'PE_coal': ('Coal', 'pef_coal'), 
                'PE_combinedgas': ('Combined Gas Cycle', 'pef_combined gas cycle'), 
                'PE_cogeneration': ('Natural Gas Cogeneration', 'pef_cogeneration'), 
                'PE_fuel': ('Fuel', 'pef_fuel'), 
                'PE_wind': ('Wind', 'pef_wind'), 
                'PE_photovoltaic': ('Photovoltaic', 'pef_photovoltaic'), 
                'PE_solarthermal': ('Solar thermal', 'pef_solar thermal'),
                'PE_oceangeothermal': ('Ocean/Geothermal', 'pef_ocean/geothermal'), 
                'PE_biomass': ('Biomass', 'pef_biomass'),
                'PE_biogas': ('Biogas', 'pef_biogas'), 
                'PE_waste': ('Waste RSU', 'pef_waste'), 
                'PE_hydroUGH': ('Hydro', 'pef_hydro'),
                'PE_pumpedhydro': ('Pumped hydro', 'pef_pumped hydro'),
                'PE_france': ('Import France', 'pef_france')
                }

    PE_df = df.copy(deep=False)
    for column, data in mapping_cfs.items():
        PE_df[column] = PE_df[data[0]] * PEF[data[1]]

    PE_df['Total PE USE'] = PE_df.drop(['Total generation', 
                                            'Balearic link',
                                            'Interconnections exchange',
                                            'Peninsular demand', 
                                            'Total generation', 
                                            'Nuclear', 
                                            'Pumped hydro',
                                            'Combined Gas Cycle', 
                                            'Photovoltaic', 
                                            'Solar thermal',
                                            'Ocean/Geothermal', 
                                            'Fuel', 
                                            'Biomass', 
                                            'Biogas', 
                                            'Hydro', 
                                            'Wind',
                                            'Natural Gas Cogeneration', 
                                            'Waste RSU', 
                                            'Coal',
                                            'Import France',
                                            'Import Portugal', 
                                            'Renewables',
                                            'NonRenewables', 
                                            'Total', 
                                            'RES-E-RATIO',
                                            'Pump Consumption',
                                            'Balearic link',
                                            'Morocco Exchange',
                                            'Portugal Exchange',
                                            'France Exchange',
                                            'Interconnections exchange'], 
                                           axis=1).sum(axis=1)

    PE_df['APEF'] = PE_df['Total PE USE']/PE_df['Peninsular demand']
    
    #df['APEF'] = PE_df['APEF']
    #df['Total PE USE'] = PE_df['Total PE USE']   
    
    
    return PE_df 

def marginal_signals(df):
    marginal = df[['Total generation', 'Peninsular demand']]
    res = df[['RES-E-RATIO']]
    co2 = df[['Total Emisions','AEF']]
    pe = df[['Total PE USE','APEF']]

    marginal_df = pd.concat([marginal, co2, pe, res], axis=1).fillna(0) 
    marginal_df["DeltaLoad"] = marginal_df["Peninsular demand"].diff()
    marginal_df["DeltaCO2"] = (marginal_df["Peninsular demand"]*marginal_df['AEF']).diff()
    marginal_df["DeltaPE"] = (marginal_df["Peninsular demand"]*marginal_df['APEF']).diff()
    marginal_df = marginal_df.fillna(0)
    #marginal_df.to_csv('marginal_df.csv')
    
    marginal_df['TGC'] = marginal_df['Peninsular demand']/1000
    marginal_df['RES'] = marginal_df['RES-E-RATIO']
    marginal_df['MEFmodel'] = numpy.nan
    marginal_df['MPEFmodel'] = numpy.nan
    # MEF coefficients
    a = [0.5705,-1.2236,0.0054,-0.0335,-0.00036,0.0300]
    for i in range(0,len(df)):
        load=marginal_df.iloc[i].TGC
        res=marginal_df.iloc[i].RES
        MEFmodelled=a[0]+a[1]*res+a[2]*load+a[3]*res*res+a[4]*load*load+a[5]*load*res
        marginal_df.MEFmodel.iloc[i]=MEFmodelled
    
    
    # MPEF coefficients
    b = [-1.6378,-1.1774,0.22068,-1.2180,-0.00384,0.02965]
    for i in range(0,len(df)):
        load=marginal_df.iloc[i].TGC
        res=marginal_df.iloc[i].RES
        MPEFmodelled=b[0]+b[1]*res+b[2]*load+b[3]*res*res+b[4]*load*load+b[5]*load*res
        marginal_df.MPEFmodel.iloc[i]=MPEFmodelled
    
    #df.to_csv('penalty_signals.csv')
    
    return marginal_df 

####### ROUTES ########

@app.route('/', methods=['POST','GET'])
def api():
    #main
    df = pd.DataFrame(columns=["PBF_shortname","PBF_value", "PBF_datetime"])
    values_df = get_values(df)
    datetime_df = set_datetime_index(values_df)
    sorted_df = pivot_sort_and_fill(datetime_df)
    rename_df = rename_en(sorted_df)
    all_df = build_df(rename_df)
    analysis_df = renewables(all_df)
    co2_df = average_carbon_emissions(analysis_df)
    pe_df = average_primary_energy(co2_df)
    marginal_df = marginal_signals(pe_df)

    result_signals = marginal_df
    result_sources = all_df.filter(['Balearic link','Nuclear','Pumped hydro','Combined Gas Cycle','Photovoltaic','Solar thermal','Ocean/Geothermal',
                                            'Fuel','Biomass', 'Biogas','Hydro', 'Wind','Natural Gas Cogeneration', 'Waste RSU',
                                            'Coal','Import France','Import Portugal','Pump Consumption','Morocco Exchange','Portugal Exchange','France Exchange']) 
        
        
    chart_from_python=two_y_axis_dropdown(result_signals)
    chart_from_python_res=pie_subplots(analysis_df)
    char_from_python_sources=my_plot_full_bar(result_sources)
    
    download_df = marginal_df.filter(['datetime','AEF','APEF','MEFmodel','MPEFmodel'])

    json_object = download_df.to_json()
    with open("static/css\day-ahead.json", "w") as outfile:
        outfile.write(json_object) 

    return render_template('GUI.html',
                           chart_for_html=chart_from_python,
                           chart_for_html_res=chart_from_python_res,
                           chart_for_html_sources=char_from_python_sources,
                           json_object_html=json_object
                           )

@app.route('/dashboard',methods=['POST','GET'])
def dashboard():
    df = pd.DataFrame(columns=["PBF_shortname","PBF_value", "PBF_datetime"])
    values_df = get_values(df)
    datetime_df = set_datetime_index(values_df)
    sorted_df = pivot_sort_and_fill(datetime_df)
    rename_df = rename_en(sorted_df)
    all_df = build_df(rename_df)
    analysis_df = renewables(all_df)
    
    result = analysis_df
    chart_from_python=pie_subplots(result)
    
    co2_df = average_carbon_emissions(analysis_df)
    pe_df = average_primary_energy(co2_df)
    marginal_df = marginal_signals(pe_df)
    
    result_co2 = co2_df.filter(['CO2_nuclear', 'CO2_coal', 'CO2_combinedgas',
       'CO2_cogeneration', 'CO2_fuel', 'CO2_wind', 'CO2_photovoltaic',
       'CO2_solarthermal', 'CO2_biomass', 'CO2_biogas', 'CO2_waste',
       'CO2_hydroUGH', 'CO2_pumpedhydro', 'CO2_france', 'CO2_portugal'])
    
    result_pe = pe_df.filter(['PE_nuclear', 'PE_coal', 'PE_combinedgas',
       'PE_cogeneration', 'PE_fuel', 'PE_wind', 'PE_photovoltaic',
       'PE_solarthermal', 'PE_biomass', 'PE_biogas', 'PE_waste', 'PE_hydroUGH',
       'PE_pumpedhydro', 'PE_france'])
    
    chart_from_python_co2=my_plot_full_bar_CO2(result_co2)
    chart_from_python_pe=my_plot_full_bar_PE(result_pe)

    return render_template('Dashboard.html',
                           chart_for_html=chart_from_python,
                            chart_for_html_co2=chart_from_python_co2,
                           chart_for_html_pe=chart_from_python_pe
                           )

Dataset = pd.read_csv('signals16_21.csv', encoding="UTF-8")

@app.route('/historical')
#methods=['GET','POST']
def historical():
    
    return render_template('historical.html')
  
@app.route('/checkhistorical')
def checkhistorical():
    start_date = req.args.get('start_date')
    #start_date = input('Introduce date: ')
    end_date = req.args.get('end_date')
    #end_date = input('Introduce date: ')
    start_date = str(start_date)
    end_date = str(end_date)
    if start_date and end_date in Dataset.datetime.values:
        print("The date is in list")
        mask = (Dataset.datetime >= start_date) & (Dataset.datetime <= end_date)
        filtered_df = Dataset.loc[mask]
        response = filtered_df
        response = filtered_df.to_json()
        print(response)
        csv = filtered_df.to_csv('historical.csv')

        json_object = filtered_df.to_json()

        with open("static/css\historical.json", "w") as outfile:
            outfile.write(json_object) 
            
        #print(filtered_df.head())
        #response = filtered_df.to_html(classes="table table-striped")
    else:
        response = ['Requested datetime series is not available.']
        print('olala')
    #return flask.jsonify({'status':True(start_date, end_date)})
    #return jsonify(response)
    return response
#app.run()
