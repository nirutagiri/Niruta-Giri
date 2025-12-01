# train.py
import argparse
import os
import numpy as np
from utils import load_csv, handle_missing, scale_features, create_sequences, save_scaler
from model import build_lstm_model, save_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def main(args):
    df = load_csv(args.csv)
    features = args.features.split(',')
    target = args.target

    # handle missing
    df = handle_missing(df, strategy=args.missing, cols=features + [target])
    X_raw = df[features].values.astype(float)
    y_raw = df[[target]].values.astype(float)

    X_scaled, scaler_X = scale_features(X_raw)
    y_scaled, scaler_y = scale_features(y_raw)

    # save scalers
    os.makedirs(args.outdir, exist_ok=True)
    save_scaler(scaler_X, os.path.join(args.outdir, 'scaler_X.joblib'))
    save_scaler(scaler_y, os.path.join(args.outdir, 'scaler_y.joblib'))

    # sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, args.seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=args.test_size, shuffle=False)

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), units=args.units, dropout=args.dropout, output_dim=1, task='regression')

    checkpoint = ModelCheckpoint(os.path.join(args.outdir, 'best_model.h5'), save_best_only=True, monitor='val_loss')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_split=args.val_split, epochs=args.epochs, batch_size=args.batch_size, callbacks=[es, checkpoint], verbose=1)

    # final save
    save_model(model, os.path.join(args.outdir, 'final_model.h5'))
    print("Training complete. Models & scalers saved to:", args.outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--features', default='Netflow_Bytes,BTC,USD')
    p.add_argument('--target', default='Prediction')
    p.add_argument('--seq_length', type=int, default=10)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--units', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--missing', choices=['drop','ffill','zero'], default='drop')
    p.add_argument('--outdir', default='saved_models')
    args = p.parse_args()
    main(args)
