from ..infrastructure.llm.graph import main_graph

try:
    png_data = main_graph.get_graph().draw_mermaid_png()
    
    with open("graph_visualization.png", "wb") as f:
        f.write(png_data)
    
    print("Graph saved to graph_visualization.png")
    
    try:
        from IPython.display import Image, display
        display(Image(png_data))
    except ImportError:
        pass
        
except Exception as e:

    print(f"Could not render graph: {e}")
    pass
