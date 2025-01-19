import matplotlib.pyplot as plt
import joblib
import os

def plot_spectra(spectra, labels, num_spectra=10, save_path=None):
    plt.figure(figsize=(10, 6))
    for i in range(min(num_spectra, len(spectra))):
        plt.plot(spectra[i], label=labels[i])
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title('Preprocessed Spectra')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    preprocessed_dir = './preprocessed_spectra'  # Directory where preprocessed data is saved
    
    # Load preprocessed spectra
    X_path = os.path.join(preprocessed_dir, 'X.pkl')
    y_path = os.path.join(preprocessed_dir, 'y.pkl')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Error: Preprocessed files not found in {preprocessed_dir}")
        return
    
    print(f"Loading preprocessed spectra from {preprocessed_dir}...")
    X = joblib.load(X_path)
    y = joblib.load(y_path)
    
    print(f"Loaded {len(X)} spectra.")
    
    # Plot the preprocessed spectra
    plot_spectra(X, y, num_spectra=10, save_path=os.path.join(preprocessed_dir, 'preprocessed_spectra.png'))

if __name__ == "__main__":
    main()