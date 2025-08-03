import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_function_dataflow_graph():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Color scheme for different files
    colors = {
        'trainer': '#FFE6CC',      # model_trainer.py functions
        'predictor': '#E8F4FD',    # predictor.py functions  
        'app': '#E8F5E8',          # app.py functions
        'data': '#F0E6FF',         # data files
        'util': '#FFF2E6'          # utility functions
    }
    
    # Function nodes with positions
    functions = [
        # model_trainer.py functions
        ('load_data', 1, 12, colors['trainer']),
        ('preprocess_text', 3, 11, colors['trainer']),
        ('extract_features', 5, 11, colors['trainer']),
        ('prepare_dataset', 3, 10, colors['trainer']),
        ('train_model', 1, 9, colors['trainer']),
        ('evaluate_model', 3, 9, colors['trainer']),
        ('save_model', 5, 9, colors['trainer']),
        ('print_metrics', 5, 8, colors['trainer']),
        
        # predictor.py functions
        ('load_saved_model', 8, 11, colors['predictor']),
        ('predict', 10, 10, colors['predictor']),
        ('prediction_loop', 8, 9, colors['predictor']),
        
        # app.py functions
        ('initialize_predictor', 1, 6, colors['app']),
        ('read_output', 3, 5, colors['app']),
        ('write_input', 5, 5, colors['app']),
        ('home', 1, 4, colors['app']),
        ('predict_spam', 3, 4, colors['app']),
        ('api_predict', 5, 4, colors['app']),
        
        # Data files
        ('spam.csv', 1, 13, colors['data']),
        ('model.ubj', 8, 13, colors['data']),
        ('vectorizer.ubj', 10, 13, colors['data']),
    ]
    
    # Draw function nodes
    node_positions = {}
    for name, x, y, color in functions:
        if name.endswith('.csv') or name.endswith('.ubj'):
            # Data files - rectangular
            box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='black', linewidth=1.5)
        else:
            # Functions - elliptical
            box = FancyBboxPatch((x-0.5, y-0.25), 1, 0.5, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold')
        node_positions[name] = (x, y)
    
    # Define connections (from_function, to_function, label)
    connections = [
        # Training flow
        ('spam.csv', 'load_data', 'read'),
        ('load_data', 'preprocess_text', 'messages'),
        ('load_data', 'extract_features', 'messages'),  
        ('preprocess_text', 'prepare_dataset', 'clean_text'),
        ('extract_features', 'prepare_dataset', 'features'),
        ('prepare_dataset', 'train_model', 'X_train, y_train'),
        ('prepare_dataset', 'evaluate_model', 'X_test, y_test'),
        ('train_model', 'save_model', 'model'),
        ('train_model', 'evaluate_model', 'model'),
        ('evaluate_model', 'print_metrics', 'metrics'),
        ('save_model', 'model.ubj', 'write'),
        ('save_model', 'vectorizer.ubj', 'write'),
        
        # Prediction flow
        ('model.ubj', 'load_saved_model', 'read'),
        ('vectorizer.ubj', 'load_saved_model', 'read'),
        ('load_saved_model', 'predict', 'model, vectorizer'),
        ('predict', 'prediction_loop', 'result'),
        ('preprocess_text', 'predict', 'clean_text'),
        ('extract_features', 'predict', 'features'),
        
        # Web app flow
        ('initialize_predictor', 'read_output', 'spawn'),
        ('initialize_predictor', 'write_input', 'spawn'),
        ('api_predict', 'write_input', 'queue.put'),
        ('read_output', 'api_predict', 'queue.get'),
        ('home', 'predict_spam', 'redirect'),
    ]
    
    # Draw connections
    for from_func, to_func, label in connections:
        if from_func in node_positions and to_func in node_positions:
            from_pos = node_positions[from_func]
            to_pos = node_positions[to_func]
            
            # Create arrow
            arrow = ConnectionPatch(from_pos, to_pos, "data", "data",
                                  arrowstyle="->", shrinkA=25, shrinkB=25, 
                                  mutation_scale=15, fc="black", lw=1.5)
            ax.add_patch(arrow)
            
            # Add label on arrow
            mid_x = (from_pos[0] + to_pos[0]) / 2
            mid_y = (from_pos[1] + to_pos[1]) / 2
            
            # Offset label slightly to avoid overlap
            offset_x = 0.1 if to_pos[0] > from_pos[0] else -0.1
            offset_y = 0.1 if to_pos[1] > from_pos[1] else -0.1
            
            ax.text(mid_x + offset_x, mid_y + offset_y, label, ha='center', va='center', 
                   fontsize=6, style='italic', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Add file groupings
    # model_trainer.py group
    trainer_group = FancyBboxPatch((0.2, 7.5), 5.8, 5.2, boxstyle="round,pad=0.1",
                                  facecolor='none', edgecolor='orange', linewidth=2, linestyle='--')
    ax.add_patch(trainer_group)
    ax.text(3, 7.7, 'model_trainer.py', ha='center', va='center', fontsize=12, 
           fontweight='bold', color='orange')
    
    # predictor.py group  
    predictor_group = FancyBboxPatch((7.2, 8.5), 3.8, 3, boxstyle="round,pad=0.1",
                                   facecolor='none', edgecolor='blue', linewidth=2, linestyle='--')
    ax.add_patch(predictor_group)
    ax.text(9, 8.7, 'predictor.py', ha='center', va='center', fontsize=12, 
           fontweight='bold', color='blue')
    
    # app.py group
    app_group = FancyBboxPatch((0.2, 3.5), 5.8, 3, boxstyle="round,pad=0.1",
                              facecolor='none', edgecolor='green', linewidth=2, linestyle='--')
    ax.add_patch(app_group)
    ax.text(3, 3.7, 'app.py', ha='center', va='center', fontsize=12, 
           fontweight='bold', color='green')
    
    # Data files group
    data_group = FancyBboxPatch((0.2, 12.5), 10.8, 1, boxstyle="round,pad=0.1",
                               facecolor='none', edgecolor='purple', linewidth=2, linestyle='--')
    ax.add_patch(data_group)
    ax.text(5.5, 12.2, 'Data Files', ha='center', va='center', fontsize=12, 
           fontweight='bold', color='purple')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=colors['trainer'], label='model_trainer.py functions'),
        mpatches.Patch(color=colors['predictor'], label='predictor.py functions'),
        mpatches.Patch(color=colors['app'], label='app.py functions'),
        mpatches.Patch(color=colors['data'], label='Data files'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Title
    ax.text(7, 13.7, 'SMS Spam Identifier - Function-Level Dataflow', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add process flow annotations
    ax.text(12, 11, 'TRAINING\nPIPELINE', ha='center', va='center', fontsize=10, 
           fontweight='bold', color='orange', rotation=0,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.text(12, 9, 'PREDICTION\nPIPELINE', ha='center', va='center', fontsize=10, 
           fontweight='bold', color='blue', rotation=0,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
           
    ax.text(12, 5, 'WEB SERVER\nPIPELINE', ha='center', va='center', fontsize=10, 
           fontweight='bold', color='green', rotation=0,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('function_dataflow.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_function_dataflow_graph()