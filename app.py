"""
Flask API para XGBoost - Classificação de Tipo de Pokémon
Modelo treinado com dataset Pokémon REAL do Kaggle
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import kagglehub

app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURAÇÃO DO MODELO
# ============================================================================

MODEL_PATH = 'xgboost_pokemon_model.pkl'
METADATA_PATH = 'pokemon_metadata.pkl'
DATASET_PATH = None

def download_pokemon_dataset():
    """Download do dataset Pokémon do Kaggle"""
    global DATASET_PATH
    
    print("Baixando dataset do Kaggle...")
    try:
        # Download do dataset
        DATASET_PATH = kagglehub.dataset_download("abcsds/pokemon")
        print(f"Dataset baixado em: {DATASET_PATH}")
        
        # Procurar o arquivo CSV
        csv_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.csv')]
        if csv_files:
            csv_path = os.path.join(DATASET_PATH, csv_files[0])
            print(f"Arquivo CSV encontrado: {csv_path}")
            return csv_path
        else:
            print("Nenhum arquivo CSV encontrado!")
            return None
    except Exception as e:
        print(f"Erro ao baixar dataset: {e}")
        print("Certifique-se de que:")
        print("1. Tem kagglehub instalado: pip install kagglehub")
        print("2. Fez login no Kaggle: kagglehub.login()")
        print("3. Tem acesso ao dataset: abcsds/pokemon")
        return None

def load_pokemon_data():
    """Carrega o dataset do Pokémon"""
    # Tenta encontrar o CSV localmente primeiro
    local_paths = [
        'Pokemon.csv',
        'pokemon.csv',
        os.path.join(DATASET_PATH, 'Pokemon.csv') if DATASET_PATH else None,
    ]
    
    csv_path = None
    for path in local_paths:
        if path and os.path.exists(path):
            csv_path = path
            break
    
    # Se não encontrar localmente, baixa do Kaggle
    if not csv_path:
        csv_path = download_pokemon_dataset()
    
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError("Dataset não encontrado. Baixe em https://www.kaggle.com/abcsds/pokemon")
    
    print(f"Carregando dataset de: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Exibir informações do dataset
    print(f"Shape: {df.shape}")
    print(f"Colunas: {df.columns.tolist()}")
    print(f"\nPrimeiras linhas:")
    print(df.head())
    
    return df

def train_model():
    """Treina o modelo XGBoost com dataset Pokémon REAL"""
    print("\n" + "="*60)
    print("TREINANDO MODELO XGBOOST COM DATASET POKÉMON")
    print("="*60)
    
    # Carregar dados
    df = load_pokemon_data()
    
    # Limpar dados
    print("\nLimpando dados...")
    
    # Remover linhas com valores faltantes nas colunas necessárias
    required_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Type 1']
    df = df.dropna(subset=required_cols)
    
    print(f"Total de Pokémons após limpeza: {len(df)}")
    print(f"Tipos: {df['Type 1'].unique()}")
    print(f"Distribuição de tipos:\n{df['Type 1'].value_counts()}\n")
    
    # Features: HP, Attack, Defense, SpA, SpD, Speed
    X = df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].values
    
    # Target: Type 1
    y = df['Type 1'].values
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Codificar targets
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Classes: {le.classes_}")
    print(f"Número de classes: {len(le.classes_)}")
    
    # Dividir dados (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTreino: {len(X_train)} amostras")
    print(f"Teste: {len(X_test)} amostras")
    
    # Criar e treinar modelo
    print("\nTreinando XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    print("Treinamento concluído!")
    
    # Avaliação
    print("\nAvaliando modelo...")
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'num_classes': len(le.classes_),
        'total_pokemon': len(df)
    }
    
    print(f"\n{'='*60}")
    print("RESULTADOS DO TREINAMENTO")
    print(f"{'='*60}")
    print(f"Acurácia:   {metrics['accuracy']:.4f}")
    print(f"Precisão:   {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-Score:   {metrics['f1']:.4f}")
    print(f"{'='*60}\n")
    
    # Salvar modelo
    joblib.dump(model, MODEL_PATH)
    joblib.dump({
        'label_encoder': le,
        'feature_names': ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'],
        'pokemon_types': list(le.classes_),
        'metrics': metrics
    }, METADATA_PATH)
    
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Metadados salvos em: {METADATA_PATH}")
    
    return model, le, metrics

def load_model():
    """Carrega o modelo treinado"""
    if not os.path.exists(MODEL_PATH):
        print("Modelo não encontrado. Treinando novo modelo...")
        model, le, metrics = train_model()
        return model
    print("Modelo carregado do arquivo.")
    return joblib.load(MODEL_PATH)

def get_metadata():
    """Obtém metadados do modelo"""
    if not os.path.exists(METADATA_PATH):
        return None
    return joblib.load(METADATA_PATH)

# ============================================================================
# ROTAS DA API
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Verifica saúde da API"""
    return jsonify({'status': 'ok', 'message': 'API rodando'})

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Retorna informações sobre o modelo"""
    metadata = get_metadata()
    if metadata is None:
        return jsonify({'error': 'Modelo não treinado'}), 400
    
    return jsonify({
        'algorithm': 'XGBoost',
        'dataset': 'Pokémon Dataset (Kaggle - abcsds/pokemon)',
        'features': metadata['feature_names'],
        'types': metadata['pokemon_types'],
        'metrics': metadata['metrics']
    })

@app.route('/api/model/train', methods=['POST'])
def train_endpoint():
    """Treina o modelo"""
    try:
        model, le, metrics = train_model()
        return jsonify({
            'message': 'Modelo treinado com sucesso',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Realiza predição com os dados fornecidos"""
    try:
        data = request.json
        
        if 'features' not in data:
            return jsonify({'error': 'Campo "features" obrigatório'}), 400
        
        features = data['features']
        
        if len(features) != 6:
            return jsonify({
                'error': 'Esperado 6 features: [HP, Attack, Defense, Sp. Atk, Sp. Def, Speed]'
            }), 400
        
        X = np.array([features])
        
        model = load_model()
        metadata = get_metadata()
        
        # Fazer predição
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Mapear classe
        pokemon_type = metadata['pokemon_types'][int(prediction)]
        
        # Organizar probabilidades
        type_probs = {}
        for i, ptype in enumerate(metadata['pokemon_types']):
            type_probs[ptype] = float(probabilities[i])
        
        # Ordenar por probabilidade
        sorted_probs = dict(sorted(type_probs.items(), key=lambda x: x[1], reverse=True))
        
        return jsonify({
            'prediction': int(prediction),
            'type': pokemon_type,
            'probabilities': sorted_probs,
            'confidence': float(max(probabilities)),
            'top_3': dict(list(sorted_probs.items())[:3])
        })
    
    except ValueError as e:
        return jsonify({'error': f'Erro de validação: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Realiza múltiplas predições"""
    try:
        data = request.json
        
        if 'samples' not in data:
            return jsonify({'error': 'Campo "samples" obrigatório'}), 400
        
        samples = data['samples']
        
        if not isinstance(samples, list) or len(samples) == 0:
            return jsonify({'error': 'samples deve ser uma lista não vazia'}), 400
        
        X = []
        for sample in samples:
            if len(sample) != 6:
                return jsonify({
                    'error': 'Cada amostra deve ter 6 features'
                }), 400
            X.append(sample)
        
        X = np.array(X)
        
        model = load_model()
        metadata = get_metadata()
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            type_probs = {}
            for j, ptype in enumerate(metadata['pokemon_types']):
                type_probs[ptype] = float(probs[j])
            
            results.append({
                'sample_id': i,
                'prediction': int(pred),
                'type': metadata['pokemon_types'][int(pred)],
                'probabilities': type_probs,
                'confidence': float(max(probs))
            })
        
        return jsonify({'results': results, 'total': len(results)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features/description', methods=['GET'])
def features_description():
    """Retorna descrição das features esperadas"""
    return jsonify({
        'features': [
            {
                'name': 'hp',
                'description': 'Hit Points (Pontos de Vida)',
                'typical_range': [1, 255]
            },
            {
                'name': 'attack',
                'description': 'Ataque Físico',
                'typical_range': [5, 165]
            },
            {
                'name': 'defense',
                'description': 'Defesa Física',
                'typical_range': [5, 250]
            },
            {
                'name': 'sp_atk',
                'description': 'Ataque Especial',
                'typical_range': [20, 154]
            },
            {
                'name': 'sp_def',
                'description': 'Defesa Especial',
                'typical_range': [20, 250]
            },
            {
                'name': 'speed',
                'description': 'Velocidade',
                'typical_range': [5, 180]
            }
        ]
    })

# ============================================================================
# INICIALIZAÇÃO
# ============================================================================

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("\n🔴 PRIMEIRO ACESSO - TREINANDO MODELO 🔴\n")
        try:
            train_model()
        except FileNotFoundError as e:
            print(f"\n⚠️  ERRO: {e}")
            print("\nPARA USAR O DATASET DO KAGGLE, FAÇA:")
            print("1. pip install kagglehub")
            print("2. kagglehub.login()  (na pasta do projeto)")
            print("3. Coloque seu Pokemon.csv na pasta do projeto OU")
            print("4. Execute este script novamente")
    
    print("\n🎮 Iniciando servidor Flask...")
    print("📍 http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)