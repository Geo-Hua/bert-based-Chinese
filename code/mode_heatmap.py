import webbrowser
import pandas as pd
import branca.colormap as cm
import folium
from folium.plugins import HeatMap
import requests

# read csv
file_path = '../data/emotion_prediction_wh.csv'
data = pd.read_csv(file_path)

#The cate category of the read
num=100
# create folium
base_map = folium.Map(
    location=[data['lat'].mean(), data['lon'].mean()],
    zoom_start=10,
    tiles='CartoDB.DarkMatter',
    control_scale=True,
    attr="Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL."
)

#  GeoJSON
wuhan_geojson = requests.get('https://geojson.cn/api/china/420000.json').json()
folium.GeoJson(
    wuhan_geojson,
    style_function=lambda x: {'color': 'white', 'weight': 1.5},
    name="武汉边界"
).add_to(base_map)

# Filtering specific types of data
filtered_data = data[data['cate'] == num]

heat_data = [[row['lat'], row['lon']] for index, row in filtered_data.iterrows()]

# Creating colour maps (but not adding them directly to the map)
colormap = cm.LinearColormap(['blue', 'lime', 'red'], vmin=0, vmax=1)
gradient_map = {0: 'blue', 0.5: 'lime', 1: 'red'}

# heatmap
HeatMap(heat_data, radius=15, blur=10, max_zoom=1, gradient=gradient_map).add_to(base_map)

# Creating HTML legends manually
legend_html = f"""
<div style="
    position: fixed;
    bottom: 50px;
    right: 50px;
    width: 250px;
    height: 80px;
    background-color: rgba(0, 0, 0, 0.7);
    z-index:9999;
    font-size:14px;
    padding: 10px;
    border-radius: 5px;
    color: white;
">
    <b>Density</b><br>
    <div style="background: linear-gradient(to right, blue, lime, red); height: 10px; width: 100%;"></div>
    <div style="display: flex; justify-content: space-between;">
        <span>Low</span>
        <span>High</span>
    </div>
</div>
"""
# Add the legend to the map as HTML
base_map.get_root().html.add_child(folium.Element(legend_html))
folium.Rectangle(
    bounds=[(data['lat'].min()-0.08, data['lon'].min()-0.2), (data['lat'].max()+0.08, data['lon'].max()+0.4)],
    color='white',
    # color=color,
    weight=1,
    fill_opacity=0.5
).add_to(base_map)
# save
output_file = f'../result/bert/wh/heatmap_by_category_{num}.html'
base_map.save(output_file)
# webbrowser.open(output_file)

print("Heatmap has been generated and saved to heatmap_by_category_11.html")
