import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_cmm_vae_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors
    layer_color = '#1f77b4'
    connection_color = '#ff7f0e'
    loss_color = '#2ca02c'
    
    # Add layers
    ax.add_patch(patches.Rectangle((0.1, 0.7), 0.2, 0.2, edgecolor='black', facecolor=layer_color))
    ax.add_patch(patches.Rectangle((0.4, 0.7), 0.2, 0.2, edgecolor='black', facecolor=layer_color))
    ax.add_patch(patches.Rectangle((0.7, 0.7), 0.2, 0.2, edgecolor='black', facecolor=layer_color))
    
    ax.text(0.2, 0.85, 'Encoder', fontsize=12, va='center', ha='center', color='white')
    ax.text(0.5, 0.85, 'Latent Space', fontsize=12, va='center', ha='center', color='white')
    ax.text(0.8, 0.85, 'Decoder', fontsize=12, va='center', ha='center', color='white')

    # Add connections
    plt.arrow(0.3, 0.8, 0.1, 0, head_width=0.02, head_length=0.02, fc=connection_color, ec=connection_color)
    plt.arrow(0.6, 0.8, 0.1, 0, head_width=0.02, head_length=0.02, fc=connection_color, ec=connection_color)

    # Add loss components
    ax.add_patch(patches.Rectangle((0.1, 0.3), 0.4, 0.2, edgecolor='black', facecolor=loss_color))
    ax.text(0.3, 0.4, 'Loss Function', fontsize=12, va='center', ha='center', color='white')

    # Set limits and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Save figure
    plt.savefig('cmm_vae_architecture.png', bbox_inches='tight')
    plt.close()

plot_cmm_vae_architecture()