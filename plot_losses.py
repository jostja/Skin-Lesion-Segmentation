import os
import json
import argparse
import plotly.graph_objects as go

# Load JSON file
def generate_plots(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        train_losses = []
        val_losses = []

        if filename.endswith(".json"):
            file_path = os.path.join(input_directory, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                if "losses" in data:
                    train_losses = data["losses"]["train_losses"]
                    val_losses = data["losses"]["val_losses"]
                    epochs = list(range(1, data["num_epochs"] + 1))

                    if not train_losses:
                        print("JSON file found without the required keys.")
                        continue

                    # filename is going to specify png file
                    basename, _ = os.path.splitext(filename)

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode='lines+markers', name='Train Loss', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=epochs, y=val_losses, mode='lines+markers', name='Validation Loss', line=dict(color='red')))

                    fig.update_layout(
                        xaxis_title="Epochs",
                        yaxis_title="Loss",
                        legend=dict(
                            x=1,
                            y=1.05,
                            traceorder='normal',
                            bgcolor='rgba(255, 255, 255, 0)',
                            bordercolor='black',
                            borderwidth=1,
                            orientation='v',
                            xanchor='right',
                            yanchor='bottom'
                        ),
                        template="plotly"  # You can change this to 'plotly' for a light theme
                    )

                    fig.write_image(f'{output_directory}/train_val_losses_{basename}.png')
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="/hpi/fs00/home/konrad.pawlak/Skin-Lesion-Segmentation/results")
    parser.add_argument("--output_directory", type=str, default="/hpi/fs00/home/konrad.pawlak/Skin-Lesion-Segmentation/visualizations")
    args = parser.parse_args()

    if not os.path.isdir(args.filepath):
        print(f"Error: The provided path '{args.filepath}' is not a valid directory.")
        exit(1)

    generate_plots(args.filepath, args.output_directory)