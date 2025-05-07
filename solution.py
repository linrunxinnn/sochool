import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib.font_manager import FontProperties
import CoolProp.CoolProp as CP  # 导入CoolProp库进行热力学计算

# 设置中文字体
try:
    font = FontProperties(family='Microsoft YaHei')
except:
    font = FontProperties(family='SimHei')

# 设置随机种子以确保结果的可重复性
np.random.seed(42)

#--------------------------------------------------------------------------------------------
# 1. 定义基本数据和函数
#--------------------------------------------------------------------------------------------

# 中国主要盆地深层盐水层数据
# 数据来源：基于文献整理的中国主要盆地深层盐水层特性
# 注意：以下数据为根据相关文献估算的近似值，实际应用时需使用准确数据
basins = {
    '松辽盆地': {
        'area': 27e3,        # 面积 (km²)
        'avg_thickness': 115,  # 平均厚度 (m)
        'avg_porosity': 0.1,  # 平均孔隙度
        'avg_depth': 2250,     # 平均深度 (m)
        'avg_temperature': 80,  # 长期平均温度 (°C)
        'avg_pressure': 20,     # 长期平均压力 (MPa)
        'salinity': 85000,     # 盐度（溶解盐的浓度） (ppm)
    },
    '二连盆地': {
        'area': 11e3,
        'avg_thickness': 75,
        'avg_porosity': 0.08,
        'avg_depth': 2750,
        'avg_temperature': 71,
        'avg_pressure': 18,
        'salinity': 110000,
    },
    '鄂尔多斯盆地': {
        'area': 55e3,
        'avg_thickness': 160,
        'avg_porosity': 0.06,
        'avg_depth': 3000,
        'avg_temperature': 90,
        'avg_pressure': 25,
        'salinity': 150000,
    },
    '渤海湾盆地': {
        'area': 4e4,
        'avg_thickness': 140,
        'avg_porosity': 0.13,
        'avg_depth': 2500,
        'avg_temperature': 88,
        'avg_pressure': 23,
        'salinity': 130000,
    },
    '苏北盆地': {
        'area': 9e3,
        'avg_thickness': 90,
        'avg_porosity': 0.15,
        'avg_depth': 2000,
        'avg_temperature': 67,
        'avg_pressure': 18,
        'salinity': 75000,
    },
    '江汉盆地': {
        'area': 7e3,
        'avg_thickness': 120,
        'avg_porosity': 0.11,
        'avg_depth': 1800,
        'avg_temperature': 83,
        'avg_pressure': 23,
        'salinity': 220000,
    },
    '四川盆地': {
        'area': 8e4,
        'avg_thickness': 220,
        'avg_porosity': 0.03,
        'avg_depth': 4000,
        'avg_temperature': 98,
        'avg_pressure': 30,
        'salinity': 280000,
    },
    '柴达木盆地': {
        'area': 3e4,
        'avg_thickness': 180,
        'avg_porosity': 0.12,
        'avg_depth': 1750,
        'avg_temperature': 105,
        'avg_pressure': 27,
        'salinity': 300000,
    },
    '准噶尔盆地': {
        'area': 6e4,
        'avg_thickness': 300,
        'avg_porosity': 0.1,
        'avg_depth': 3000,
        'avg_temperature': 83,
        'avg_pressure': 23,
        'salinity': 115000,
    },
    '塔里木盆地': {
        'area': 11e4,
        'avg_thickness': 450,
        'avg_porosity': 0.05,
        'avg_depth': 4500,
        'avg_temperature': 125,
        'avg_pressure': 40,
        'salinity': 220000,
    },
    '海拉尔盆地': {
        'area': 6e3,
        'avg_thickness': 65,
        'avg_porosity': 0.12,
        'avg_depth': 2000,
        'avg_temperature': 68,
        'avg_pressure': 18,
        'salinity': 85000,
    }
}

# 将字典转换为DataFrame用于后续分析
df_basins = pd.DataFrame.from_dict(basins, orient='index')

# 使用CoolProp计算CO2的密度
def calculate_co2_density_coolprop(temperature, pressure):
    """
    使用CoolProp计算给定温度和压力下CO2的密度
    温度单位：°C
    压力单位：MPa
    密度单位：kg/m³
    """
    try:
        # 转换单位: 摄氏度→开尔文, MPa→Pa
        T_kelvin = temperature + 273.15
        P_pascal = pressure * 1e6
        
        # 使用CoolProp计算CO2的密度
        density = CP.PropsSI('D', 'T', T_kelvin, 'P', P_pascal, 'CO2')        
        return density
    except Exception as e:
        # 如果出现计算错误（如超出CoolProp有效范围），返回备用计算结果
        print(f"CoolProp计算出错: {e}. 使用备用计算方法.")
        return calculate_co2_density_backup(temperature, pressure)

# 备用CO2密度计算函数（当CoolProp出错时使用）
def calculate_co2_density_backup(temperature, pressure):
    """
    备用CO2密度计算函数
    基于简化状态方程，在CoolProp无法计算时使用
    """
    # 将温度从摄氏度转换为开尔文
    T_kelvin = temperature + 273.15
    
    # 根据温度和压力范围分段计算密度
    # 这是一个简化模型，实际应用中应使用更精确的状态方程
    if T_kelvin < 304.13:  # 临界温度
        if pressure < 7.38:  # 临界压力(MPa)
            # 气态区域
            density = 1.98 * pressure * 273.15 / T_kelvin
        else:
            # 液态区域
            density = 1000 - 1.5 * (T_kelvin - 273.15)
    else:
        # 超临界区域
        # 使用简化的经验公式
        density = 400 + 800 * (1 - np.exp(-0.1 * (pressure - 7.38))) - 2 * (T_kelvin - 304.13)
    
    return max(density, 0)  # 确保密度不为负值

# 改进的CO2-盐水密度计算函数
def calculate_co2_brine_density(temperature, pressure, salinity):
    """
    计算CO2-盐水混合系统的密度
    温度单位：°C
    压力单位：MPa
    盐度单位：ppm
    密度单位：kg/m³
    """
    # 计算CO2密度
    co2_density = calculate_co2_density_coolprop(temperature, pressure)
    
    # 计算盐水密度（简化模型）
    # 将盐度从ppm转换为质量分数
    salinity_mass_fraction = salinity / 1e6
    
    # 纯水密度（随温度变化）
    water_density = 1000 * (1 - ((temperature - 4) / 1000) ** 2)
    
    # 盐水密度 = 纯水密度 * (1 + 系数 * 盐度质量分数)
    # 这是一个简化模型，实际应用中应使用更精确的计算方法
    brine_density = water_density * (1 + 0.7 * salinity_mass_fraction)
    
    # 由于CO2和盐水基本不混溶，这里不考虑混合效应
    # 实际情况下，应考虑互溶性和界面效应
    
    return co2_density, brine_density

# 改进的孔隙中CO2可占比例的估算函数
def estimate_co2_efficiency_factor(depth, porosity, temperature, pressure, salinity):
    """
    估算CO2在孔隙中可占的比例（效率因子）
    考虑深度、孔隙度、温度、压力和盐度的综合影响
    """
    # 基础效率因子（行业经验值通常在0.2-0.6之间）
    base_factor = 0.35
    
    # 深度调整（深度越大，压力越大，CO2越容易压缩进孔隙）
    depth_adjustment = 0.08 * np.tanh(depth / 3000)
    
    # 孔隙度调整（孔隙度越高，CO2占比可能越大）
    porosity_adjustment = 0.1 * np.tanh(porosity / 0.15)
    
    # 温度和压力调整
    # 温度高时CO2流动性好，但同时水也更容易占据孔隙
    temp_adjustment = 0.03 * np.tanh((temperature - 60) / 30)
    
    # 压力高有利于CO2注入
    pressure_adjustment = 0.05 * np.tanh((pressure - 20) / 10)
    
    # 盐度调整（盐度高会影响CO2溶解度和流动性）
    salinity_adjustment = 0.02 * np.tanh((salinity - 100000) / 50000)
    
    # 计算综合效率因子
    efficiency_factor = base_factor + depth_adjustment + porosity_adjustment + temp_adjustment + pressure_adjustment + salinity_adjustment
    
    # 确保效率因子在合理范围内 (0.2 - 0.7)
    return max(0.2, min(0.7, efficiency_factor))

#--------------------------------------------------------------------------------------------
# 2. 计算总孔隙体积和CO2储量
#--------------------------------------------------------------------------------------------

def calculate_total_storage_capacity():
    """
    计算中国各主要盆地深层盐水层的CO2储存容量
    返回总储量（单位：亿吨）和各盆地详细数据
    """
    results = {}
    total_capacity_gt = 0  # 总容量（单位：亿吨）
    
    for basin_name, basin_data in basins.items():
        # 计算总孔隙体积 (m³)
        area_m2 = basin_data['area'] * 1e6  # 将km²转换为m²
        thickness_m = basin_data['avg_thickness']
        porosity = basin_data['avg_porosity']
        
        total_pore_volume = area_m2 * thickness_m * porosity

        print(f"盆地: {basin_name}, 总孔隙体积: {total_pore_volume:.2f} m³")
        
        # 计算CO2密度和盐水密度 (kg/m³)
        co2_density, brine_density = calculate_co2_brine_density(
            basin_data['avg_temperature'], 
            basin_data['avg_pressure'],
            basin_data['salinity']
        )
        # print(f"CO2密度: {co2_density:.2f} kg/m³, 盐水密度: {brine_density:.2f} kg/m³")
        
        # 估算效率因子（CO2在孔隙中可占的比例）
        efficiency_factor = estimate_co2_efficiency_factor(
            basin_data['avg_depth'], 
            porosity,
            basin_data['avg_temperature'],
            basin_data['avg_pressure'],
            basin_data['salinity']
        )
        
        # 计算CO2储存容量 (kg)
        co2_storage_capacity_kg = total_pore_volume * efficiency_factor * co2_density
        
        # 转换为Gt (亿吨)
        co2_storage_capacity_gt = 0.05 * co2_storage_capacity_kg / 1e11
        
        # 存储结果
        results[basin_name] = {
            'total_pore_volume_km3': total_pore_volume / 1e9,  # 转换为km³
            'co2_density_kg_m3': co2_density,
            'brine_density_kg_m3': brine_density,
            'efficiency_factor': efficiency_factor,
            'storage_capacity_gt': co2_storage_capacity_gt
        }
        
        total_capacity_gt += co2_storage_capacity_gt
    
    return total_capacity_gt, results

#--------------------------------------------------------------------------------------------
# 3. 敏感性分析
#--------------------------------------------------------------------------------------------

def density_sensitivity_analysis():
    """
    对CO2密度进行敏感性分析
    分析不同温度和压力条件下CO2密度的变化及其对总储量的影响
    """
    # 创建温度和压力的网格
    temperatures = np.linspace(30, 120, 10)  # 30-120°C
    pressures = np.linspace(10, 40, 10)      # 10-40 MPa
    
    # 计算每个温度和压力组合的CO2密度
    densities = np.zeros((len(temperatures), len(pressures)))
    for i, temp in enumerate(temperatures):
        for j, press in enumerate(pressures):
            densities[i, j] = calculate_co2_density_coolprop(temp, press)
    
    # 创建基准案例
    base_capacity, _ = calculate_total_storage_capacity()
    
    # 分析温度变化对储量的影响
    temp_variations = np.linspace(-20, 20, 9)  # 温度变化 ±20°C
    temp_capacities = []
    
    for temp_var in temp_variations:
        # 创建一个新的盆地数据副本并修改温度
        modified_basins = {}
        for basin_name, basin_data in basins.items():
            modified_basin = basin_data.copy()
            modified_basin['avg_temperature'] = basin_data['avg_temperature'] + temp_var
            modified_basins[basin_name] = modified_basin
        
        # 计算修改后的储量
        temp_capacity = 0
        for basin_name, basin_data in modified_basins.items():
            area_m2 = basin_data['area'] * 1e6
            thickness_m = basin_data['avg_thickness']
            porosity = basin_data['avg_porosity']
            
            total_pore_volume = area_m2 * thickness_m * porosity
            co2_density, _ = calculate_co2_brine_density(
                basin_data['avg_temperature'], 
                basin_data['avg_pressure'],
                basin_data['salinity']
            )
            efficiency_factor = estimate_co2_efficiency_factor(
                basin_data['avg_depth'], 
                porosity,
                basin_data['avg_temperature'],
                basin_data['avg_pressure'],
                basin_data['salinity']
            )
            
            co2_storage_capacity_kg = total_pore_volume * efficiency_factor * co2_density
            co2_storage_capacity_gt = co2_storage_capacity_kg / 1e11
            
            temp_capacity += co2_storage_capacity_gt
        
        temp_capacities.append(temp_capacity)
    
    # 分析压力变化对储量的影响
    pressure_variations = np.linspace(-10, 10, 9)  # 压力变化 ±10 MPa
    pressure_capacities = []
    
    for pressure_var in pressure_variations:
        # 创建一个新的盆地数据副本并修改压力
        modified_basins = {}
        for basin_name, basin_data in basins.items():
            modified_basin = basin_data.copy()
            modified_basin['avg_pressure'] = max(5, basin_data['avg_pressure'] + pressure_var)  # 确保压力不小于5MPa
            modified_basins[basin_name] = modified_basin
        
        # 计算修改后的储量
        pressure_capacity = 0
        for basin_name, basin_data in modified_basins.items():
            area_m2 = basin_data['area'] * 1e6
            thickness_m = basin_data['avg_thickness']
            porosity = basin_data['avg_porosity']
            
            total_pore_volume = area_m2 * thickness_m * porosity
            co2_density, _ = calculate_co2_brine_density(
                basin_data['avg_temperature'], 
                basin_data['avg_pressure'],
                basin_data['salinity']
            )
            efficiency_factor = estimate_co2_efficiency_factor(
                basin_data['avg_depth'], 
                porosity,
                basin_data['avg_temperature'],
                basin_data['avg_pressure'],
                basin_data['salinity']
            )
            
            co2_storage_capacity_kg = total_pore_volume * efficiency_factor * co2_density
            co2_storage_capacity_gt = co2_storage_capacity_kg / 1e11
            
            pressure_capacity += co2_storage_capacity_gt
        
        pressure_capacities.append(pressure_capacity)
    
    # 分析盐度对储量的影响 (新增)
    salinity_variations = np.linspace(-50000, 50000, 9)  # 盐度变化 ±50000 ppm
    salinity_capacities = []
    
    for salinity_var in salinity_variations:
        # 创建一个新的盆地数据副本并修改盐度
        modified_basins = {}
        for basin_name, basin_data in basins.items():
            modified_basin = basin_data.copy()
            modified_basin['salinity'] = max(10000, basin_data['salinity'] + salinity_var)  # 确保盐度不小于10000ppm
            modified_basins[basin_name] = modified_basin
        
        # 计算修改后的储量
        salinity_capacity = 0
        for basin_name, basin_data in modified_basins.items():
            area_m2 = basin_data['area'] * 1e6
            thickness_m = basin_data['avg_thickness']
            porosity = basin_data['avg_porosity']
            
            total_pore_volume = area_m2 * thickness_m * porosity
            co2_density, _ = calculate_co2_brine_density(
                basin_data['avg_temperature'], 
                basin_data['avg_pressure'],
                basin_data['salinity']
            )
            efficiency_factor = estimate_co2_efficiency_factor(
                basin_data['avg_depth'], 
                porosity,
                basin_data['avg_temperature'],
                basin_data['avg_pressure'],
                basin_data['salinity']
            )
            
            co2_storage_capacity_kg = total_pore_volume * efficiency_factor * co2_density
            co2_storage_capacity_gt = co2_storage_capacity_kg / 1e11
            
            salinity_capacity += co2_storage_capacity_gt
        
        salinity_capacities.append(salinity_capacity)
    
    return {
        'temperatures': temperatures,
        'pressures': pressures,
        'densities': densities,
        'base_capacity': base_capacity,
        'temp_variations': temp_variations,
        'temp_capacities': temp_capacities,
        'pressure_variations': pressure_variations,
        'pressure_capacities': pressure_capacities,
        'salinity_variations': salinity_variations,
        'salinity_capacities': salinity_capacities
    }

#--------------------------------------------------------------------------------------------
# 4. 模型评估与可视化
#--------------------------------------------------------------------------------------------

def visualize_results(capacity_results, sensitivity_results):
    """
    可视化计算结果和敏感性分析
    """
    # 设置图表样式
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 1. 各盆地储量分布图
    basin_names = list(capacity_results.keys())
    basin_capacities = [capacity_results[basin]['storage_capacity_gt'] for basin in basin_names]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(basin_names, basin_capacities, color='skyblue')
    plt.title('中国各主要盆地深层盐水层CO2储存容量', fontproperties=font, fontsize=16)
    plt.xlabel('盆地名称', fontproperties=font, fontsize=14)
    plt.ylabel('储存容量 (亿吨)', fontproperties=font, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontproperties=font)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('basin_capacity_distribution.png', dpi=300)
    
    # 2. CO2密度热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(sensitivity_results['densities'], 
                xticklabels=[f'{p:.1f}' for p in sensitivity_results['pressures']],
                yticklabels=[f'{t:.1f}' for t in sensitivity_results['temperatures']],
                cmap='viridis', annot=False, fmt='.1f', cbar_kws={'label': '密度 (kg/m³)'})
    plt.title('不同温度和压力条件下CO2密度分布 (CoolProp计算)', fontproperties=font, fontsize=16)
    plt.xlabel('压力 (MPa)', fontproperties=font, fontsize=14)
    plt.ylabel('温度 (°C)', fontproperties=font, fontsize=14)
    plt.tight_layout()
    plt.savefig('co2_density_heatmap.png', dpi=300)
    
    # 3. 温度敏感性分析图
    plt.figure(figsize=(10, 6))
    plt.plot(sensitivity_results['temp_variations'], sensitivity_results['temp_capacities'], 
             marker='o', linestyle='-', color='red', linewidth=2)
    plt.axhline(y=sensitivity_results['base_capacity'], color='blue', linestyle='--', 
                label=f'基准储量: {sensitivity_results["base_capacity"]:.1f} 亿吨')
    plt.title('温度变化对CO2储存容量的影响', fontproperties=font, fontsize=16)
    plt.xlabel('温度变化 (°C)', fontproperties=font, fontsize=14)
    plt.ylabel('总储存容量 (亿吨)', fontproperties=font, fontsize=14)
    plt.grid(True)
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig('temperature_sensitivity.png', dpi=300)
    
    # 4. 压力敏感性分析图
    plt.figure(figsize=(10, 6))
    plt.plot(sensitivity_results['pressure_variations'], sensitivity_results['pressure_capacities'], 
             marker='s', linestyle='-', color='green', linewidth=2)
    plt.axhline(y=sensitivity_results['base_capacity'], color='blue', linestyle='--', 
                label=f'基准储量: {sensitivity_results["base_capacity"]:.1f} 亿吨')
    plt.title('压力变化对CO2储存容量的影响', fontproperties=font, fontsize=16)
    plt.xlabel('压力变化 (MPa)', fontproperties=font, fontsize=14)
    plt.ylabel('总储存容量 (亿吨)', fontproperties=font, fontsize=14)
    plt.grid(True)
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig('pressure_sensitivity.png', dpi=300)
    
    # 5. 盐度敏感性分析图 (新增)
    plt.figure(figsize=(10, 6))
    plt.plot(sensitivity_results['salinity_variations'], sensitivity_results['salinity_capacities'], 
             marker='^', linestyle='-', color='purple', linewidth=2)
    plt.axhline(y=sensitivity_results['base_capacity'], color='blue', linestyle='--', 
                label=f'基准储量: {sensitivity_results["base_capacity"]:.1f} 亿吨')
    plt.title('盐度变化对CO2储存容量的影响', fontproperties=font, fontsize=16)
    plt.xlabel('盐度变化 (ppm)', fontproperties=font, fontsize=14)
    plt.ylabel('总储存容量 (亿吨)', fontproperties=font, fontsize=14)
    plt.grid(True)
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig('salinity_sensitivity.png', dpi=300)
    
    # 6. 效率因子与深度和孔隙度的关系
    depths = np.linspace(1000, 4000, 50)
    porosities = np.linspace(0.05, 0.25, 50)
    
    # 选择一个典型的温度、压力和盐度值
    typical_temp = 75  # °C
    typical_pressure = 25  # MPa
    typical_salinity = 150000  # ppm
    
    efficiency_factors = np.zeros((len(depths), len(porosities)))
    for i, depth in enumerate(depths):
        for j, porosity in enumerate(porosities):
            efficiency_factors[i, j] = estimate_co2_efficiency_factor(
                depth, porosity, typical_temp, typical_pressure, typical_salinity)
    
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(porosities, depths)
    contour = plt.contourf(X, Y, efficiency_factors, 20, cmap='coolwarm')
    plt.colorbar(label='效率因子')
    plt.title('深度和孔隙度对CO2效率因子的影响', fontproperties=font, fontsize=16)
    plt.xlabel('孔隙度', fontproperties=font, fontsize=14)
    plt.ylabel('深度 (m)', fontproperties=font, fontsize=14)
    plt.tight_layout()
    plt.savefig('efficiency_factor_relationship.png', dpi=300)
    
    # 7. CO2密度与温度的关系曲线 (新增)
    typical_pressures = [7, 9, 10, 11, 12, 14, 20, 30, 40, 50, 60]  # MPa
    temps = np.linspace(30, 120, 100)  # °C
    
    plt.figure(figsize=(10, 6))
    for pressure in typical_pressures:
        densities = [calculate_co2_density_coolprop(temp, pressure) for temp in temps]
        plt.plot(temps, densities, label=f'压力 = {pressure} MPa', linewidth=2)
    
    plt.title('不同压力下CO2密度随温度的变化', fontproperties=font, fontsize=16)
    plt.xlabel('温度 (°C)', fontproperties=font, fontsize=14)
    plt.ylabel('CO2密度 (kg/m³)', fontproperties=font, fontsize=14)
    plt.grid(True)
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig('co2_density_temperature_curves.png', dpi=300)

#--------------------------------------------------------------------------------------------
# 5. 主程序执行
#--------------------------------------------------------------------------------------------

def main():
    """
    主程序：执行计算和可视化
    """
    print("开始计算中国深层盐水层CO2储存容量 (使用CoolProp计算CO2物性)...")
    
    # 计算储存容量
    total_capacity, basin_capacities = calculate_total_storage_capacity()
    print(f"总储存容量: {total_capacity:.2f} 亿吨CO2")
    
    # 打印各盆地详细信息
    print("\n各盆地储存容量详情:")
    for basin, data in basin_capacities.items():
        print(f"{basin}:")
        print(f"  - 总孔隙体积: {data['total_pore_volume_km3']:.2f} km³")
        print(f"  - CO2密度: {data['co2_density_kg_m3']:.2f} kg/m³")
        print(f"  - 盐水密度: {data['brine_density_kg_m3']:.2f} kg/m³")
        print(f"  - 效率因子: {data['efficiency_factor']:.2f}")
        print(f"  - 储存容量: {data['storage_capacity_gt']:.2f} 亿吨")
    
    # 进行敏感性分析
    print("\n开始CO2密度敏感性分析...")
    sensitivity_results = density_sensitivity_analysis()
    
    # 可视化结果
    print("\n生成可视化图表...")
    visualize_results(basin_capacities, sensitivity_results)
    
    print("\n计算完成，图表已保存。")
    
    # 返回计算结果供报告使用
    # 返回计算结果供报告使用
    return {
        'total_capacity': total_capacity,
        'basin_capacities': basin_capacities,
        'sensitivity_results': sensitivity_results
    }

# 执行主程序
if __name__ == "__main__":
    results = main()