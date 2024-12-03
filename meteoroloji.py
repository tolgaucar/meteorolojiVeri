from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Eğitilmiş modeli yükle
model = joblib.load('meteorolojiModel.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Modelin eğitiminde kullanılan özellik isimlerini sırasıyla belirleyelim
        feature_names = ['Hi Out', 'Low Temp', 'Dew Hum', 'Wind Pt.', 'THW Index', 
                         'Rain Index', 'Heat Bar', 'In D-D', 'In Temp', 'In Hum', 
                         'In Dew', 'Air Heat', 'Wind EMC', 'Wind Density', 'ISS Samp']
        
        # Formdan gelen verileri alalım ve modelin beklediği sıraya göre düzenleyelim
        input_features = []
        for feature in feature_names:
            input_features.append(request.form.get(feature, type=float))
        
        # Verinin doğru şekilde alındığını kontrol et
        input_features_array = np.array(input_features).reshape(1, -1)
        print("Input Features (Reshaped):", input_features_array)

        # Tahmin yapalım
        prediction = model.predict(input_features_array)
        print("Prediction Result:", prediction)

        return render_template('index.html', prediction_text=f"Tahmin Edilen Out Temp: {prediction[0]:.2f}")
    
    except ValueError as ve:
        print("Değer Hatası:", ve)
        return render_template('index.html', prediction_text="Geçersiz veri girdiniz!")
    
    except Exception as e:
        print("Hata:", e)
        return render_template('index.html', prediction_text="Bir hata oluştu. Detaylar logda.")

# Flask uygulamasını çalıştır
if __name__ == '__main__':
    app.run(debug=True)
