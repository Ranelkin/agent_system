from dataclasses import dataclass
from typing import Optional

@dataclass
class Node: 
    '''Node in Graph'''        
    label: str
    children: Optional[list] = None 
    
    