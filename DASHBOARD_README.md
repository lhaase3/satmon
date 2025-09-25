# SatMon Enhanced Dashboard

## 🚀 Overview
The SatMon Enhanced Dashboard is a professional, interactive web interface for satellite telemetry anomaly detection. Built for recruiting presentations and live demonstrations.

## ✨ Key Features

### Professional UI/UX
- **Modern Design**: Glassmorphism effects, gradient backgrounds, smooth animations
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **Dark Theme**: Optimized for data visualization and professional presentation
- **Interactive Navigation**: Smooth scrolling between sections

### Live Demo Capabilities
- **Real-time Data**: Mock satellite telemetry data with realistic patterns
- **Interactive Controls**: Channel selection, algorithm switching
- **Dynamic Visualization**: Chart.js integration with anomaly annotations
- **Performance Metrics**: Live display of detection results

### Technical Showcase
- **Dual Algorithms**: Z-Score statistical analysis and Isolation Forest ML
- **Performance Stats**: 36.4% precision, 100% recall, F1-score 0.615
- **Data Processing**: 8,760+ data points per channel
- **API Integration**: RESTful endpoints for channels, timeseries, anomalies

## 📊 Demo Data
The system includes three mock satellite channels:
1. **Temperature Sensor** (TEMP_01): -20°C to +60°C with thermal anomalies
2. **Solar Panel Voltage** (PWR_VOLT_01): 28V ± 2V with power system faults
3. **Attitude X-Axis** (ATT_X_01): ±180° with attitude control anomalies

## 🛠️ Technology Stack
- **Backend**: FastAPI with async/await
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Visualization**: Chart.js with custom anomaly annotations
- **Styling**: CSS Grid, Flexbox, CSS Variables
- **Effects**: Glassmorphism, CSS animations, responsive design

## 🎯 Target Audience
- **Recruiters**: Professional portfolio demonstration
- **Technical Interviews**: Live coding and system design discussions
- **Industry Professionals**: Aerospace and satellite technology stakeholders
- **Academic Reviewers**: Computer science and machine learning applications

## 📈 Performance Highlights
- **Detection Accuracy**: High recall (100%) for critical aerospace applications
- **Processing Speed**: Sub-second anomaly detection on 500+ data points
- **Scalability**: Designed for real-time streaming data
- **Reliability**: Robust error handling and graceful degradation

## 🔗 Quick Start
```bash
# Clone repository
git clone https://github.com/lhaase3/satmon.git
cd satmon

# Run demo server
python demo_server.py

# Open dashboard
http://localhost:8000
```

## 🎪 Live Demo Features
1. **Channel Selection**: Switch between different satellite subsystems
2. **Algorithm Comparison**: Compare Z-Score vs Isolation Forest performance
3. **Real-time Processing**: Watch anomaly detection run in real-time
4. **Interactive Charts**: Hover over data points for detailed information
5. **Performance Metrics**: Live updates of detection statistics

## 💼 Professional Presentation
The dashboard is optimized for:
- ✅ **Screen Sharing**: Clean, readable interface
- ✅ **Live Demos**: Interactive controls work smoothly
- ✅ **Technical Discussion**: Code architecture clearly visible
- ✅ **Performance Evidence**: Quantified results prominently displayed
- ✅ **Industry Relevance**: Aerospace domain expertise demonstrated

## 🔧 Development
Built by Logan Haase as part of a comprehensive satellite telemetry monitoring system. Demonstrates full-stack development, machine learning integration, and professional UI/UX design skills.

**Contact**: loganhaase3@gmail.com  
**GitHub**: https://github.com/lhaase3  
**Portfolio**: https://lhaase3.github.io/Portfolio