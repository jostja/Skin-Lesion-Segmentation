import os
import json
import argparse
import random
import plotly.graph_objects as go

def generate_colors(num_colors):
    """Generate a fixed set of unique colors based on a fixed random seed."""
    random.seed(10)  # Ensures colors remain the same across runs
    colors = []
    for _ in range(num_colors):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append(f"rgba({r},{g},{b},0.7)")
    return colors

def load_data_from_jsons(directory):
    """Load jaccard_index and threshold_jaccard_index from all JSON files in a directory."""
    jaccard_data = []
    threshold_jaccard_data = []
    dice_data = []
    filenames = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                if "test_results" in data["losses"]:
                    jaccard_data.append(data["losses"]["test_results"]["jaccard_index"])
                    threshold_jaccard_data.append(data["losses"]["test_results"]["threshold_jaccard_index"])
                    dice_data.append(data["losses"]["test_results"]["Dice_loss"])
                    basename, _ = os.path.splitext(filename)
                    filenames.append(basename)
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")

    return jaccard_data, threshold_jaccard_data, dice_data, filenames

def generate_plots(jaccard_data, threshold_jaccard_data, dice_data, filenames, output_directory):
    """Create a grouped bar chart with separate x-axis titles for Jaccard Index and Threshold Jaccard Index."""
    
    num_files = len(filenames)
    colors = generate_colors(num_files)

    fig_jacc = go.Figure()
    fig_dice = go.Figure()

    x_positions = list(range(num_files)) + [i + 2 + num_files for i in range(num_files)]

    # Add bars for Jaccard Index
    for i in range(num_files):
        fig_jacc.add_trace(go.Bar(
            x=[x_positions[i]],
            y=[jaccard_data[i]],
            marker_color=colors[i], width=0.4,
            name=filenames[i],
            showlegend=True
        ))

    # Add bars for Threshold Jaccard Index
    for i in range(num_files):
        fig_jacc.add_trace(go.Bar(
            x=[x_positions[i + num_files]],
            y=[threshold_jaccard_data[i]],
            marker_color=colors[i], width=0.4,
            showlegend=False
        ))

    # Add bars for Dice Loss
    for i in range(num_files):
        fig_dice.add_trace(go.Bar(
            x=[i+1],
            y=[dice_data[i]],
            marker_color=colors[i], width=0.4,
            name=filenames[i]  
        ))

    # Customize layout
    fig_jacc.update_layout(
        xaxis=dict(
            tickvals=[num_files / 2 - 0.5, num_files + 2 + num_files / 2 - 0.5],
            ticktext=["Jaccard Index", "Threshold Jaccard Index"]
        ),
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
        yaxis_title="Values",
        template="plotly_white",
        barmode="group"
    )

    fig_dice.update_layout(
        xaxis=dict(
            tickvals=[],
            ticktext=[""],
        ),
        legend=dict(
            x=1,
            y=1.05,
            traceorder='normal',
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='black',
            borderwidth=1,
            orientation="v",
            xanchor='right',
            yanchor='bottom'
        ),
        xaxis_title='Dice Loss',
        yaxis_title='Values',
        template='plotly_white'
    )

    # Save plot as PNG
    fig_jacc.write_image(f'{output_directory}/jaccard_comparison.png')
    fig_dice.write_image(f'{output_directory}/dice_comparison.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="/hpi/fs00/home/konrad.pawlak/Skin-Lesion-Segmentation/results")
    parser.add_argument("--output_directory", type=str, default="/hpi/fs00/home/konrad.pawlak/Skin-Lesion-Segmentation/visualizations")
    args = parser.parse_args()

    if not os.path.isdir(args.filepath):
        print(f"Error: The provided path '{args.filepath}' is not a valid directory.")
        exit(1)

    jaccard_data, threshold_jaccard_data, dice_data, filenames = load_data_from_jsons(args.filepath)

    if not jaccard_data:
        print("No valid JSON files found with the required keys.")
    else:
        generate_plots(jaccard_data, threshold_jaccard_data, dice_data, filenames, args.output_directory)
