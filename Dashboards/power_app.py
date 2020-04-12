from flask import Flask, render_template, request, redirect
import pandas as pd
import bokeh
from bokeh.embed import components
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from math import pi

from bokeh.io import output_file, show
import bokeh.palettes
from bokeh.plotting import figure
from bokeh.transform import cumsum
power_app = Flask(__name__)

# Load csv
data_csv = "https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv"
power_plant = pd.read_csv(data_csv)

# # Group by country
# power_plant = power_plant.groupby(by = 'country_long').sum()
# country_info.reset_index(inplace = True)

# Columns to be selected
country_names = power_plant['country_long'].unique().tolist()

# Create the main plot
def create_pie(_name,_df):
    x = _df[_df['country_long'] == _name]['primary_fuel'].value_counts()

    data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'fuel_type'})
    data['angle'] = data['value'] / data['value'].sum() * 2 * pi
    b = (_df[_df['country_long']==_name].groupby(by='primary_fuel').sum())['capacity_mw']/1000
    data2 = pd.Series(b).reset_index(name="capacity_mw").rename(columns={'index':'capacity_mw'})
    data2.columns = ['fuel_type', 'capacity_mw']
    data = pd.merge(data, data2, on='fuel_type')
    print(len(x))
    # Few countries have less than 3 sources of fuel
    if len(x) < 3:
        if len(x) < 2:
            data['color'] = ['#1f77b4']
        else:
            data['color'] = ['#1f77b4', '#ff7f0e']
    else:
        data['color'] = bokeh.palettes.Category20[len(x)]
    print(data)
    p = figure(plot_height=250, plot_width= 500, toolbar_location=None,
               tools="hover", tooltips="@fuel_type type: @value plants", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.3,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend='fuel_type', source=data)

    p.axis.axis_label = str(_name)
    p.axis.visible = False
    p.grid.grid_line_color = None

    return p

def create_hbar(_df, _name):

    grouped = (_df[_df['country_long']==_name].groupby('primary_fuel').sum()/1000).sort_values(by='capacity_mw', ascending=True)
    source = ColumnDataSource(grouped)
    fuels = source.data['primary_fuel'].tolist()
    p = figure(plot_height=250, plot_width= 500, toolbar_location=None,
               tools="hover",tooltips="@capacity_mw GW",y_range=fuels)
    p.hbar(y='primary_fuel', right='capacity_mw', source=source, height = 0.25, color='#ff7f0e')
    p.xaxis.axis_label = str(_name) +' Capacity (GW)'
    return p


@power_app.route("/")
def main():
    return 'Welcome'

@power_app.route("/dashboard")
def dashboard():
    current_country_name = request.args.get("country_name")
    total_capacity = round(power_plant[power_plant['country_long']==current_country_name]['capacity_mw'].sum()/1000, 2)
    num_oper = power_plant[power_plant['country_long'] == current_country_name]['name'].nunique()
    fuel_sources = power_plant[power_plant['country_long'] == current_country_name]['primary_fuel'].nunique()
    # Create the plot
    pie_plot = create_pie(current_country_name, power_plant)

    # Embed plot into HTML via Flask Render
    pie_script, pie_div = components(pie_plot)

    hbar_plot = create_hbar(power_plant, current_country_name)
    hbar_script, hbar_div = components(hbar_plot)
    #print(div)
    return render_template('dashboard.html',
                           country_names=country_names,
                           current_country_name=current_country_name,
                           total_capacity=total_capacity,
                           num_oper = num_oper,
                           fuel_sources=fuel_sources,
                           pie_script=pie_script,
                           pie_div=pie_div,
                           hbar_script=hbar_script,
                           hbar_div=hbar_div)

if __name__=='__main__':
    power_app.run(debug=True)


