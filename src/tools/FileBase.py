import os
import logging

logger = logging.getLogger('FileBase')

class FileBase():
    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.files: list[str] = []  

    def traverse(self) -> int:
        """Traverse directory and collect all files (not directories)

        Returns:
            int: number of files in directory
        """
        try: 
            stack = [self.dir]
            result = []

            while stack:
                curr = stack.pop()
                
                if os.path.isfile(curr): #Prevents adding dir to list
                    result.append(curr)
                elif os.path.isdir(curr):
                    try:
                        children = [os.path.join(curr, child) for child in os.listdir(curr)]
                        stack.extend(reversed(children))
                    except PermissionError:
                        continue
            
            result = [f for f in result if "__pycache__" not in f and ".cpython" not in f]
            
            self.files = result 
            logger.info(f'Traversed filebase. Found {len(result)} files')
            return len(result)
        except Exception as e: 
            logger.error(f"Failed to traverse FileBase: {e}")
            return 0
    
    def file_content(self, file: str) -> str:
        """Read and return file content
        
        Args:
            file: Path to file
            
        Returns:
            str: File content or None if error
        """
        try:    
            with open(file, 'r', encoding='utf-8') as f: 
                content = f.read() 
            return content
        except Exception as e: 
            logger.error(f"Failed to load File Content for {file}: {e}")
            return None  # Return None instead of nothing
    
    def update_file_content(self, file: str, updated_content: str):
        """Update file with new content
        
        Args:
            file: Path to file
            updated_content: New content to write
        """
        try:  
            f_name = os.path.basename(file)  # Better way to get filename
            with open(file, "w", encoding='utf-8') as f: 
                f.write(updated_content)
                logger.info(f'Updated file: {f_name}')
        except Exception as e: 
            logger.error(f"Failed to update File Content: {e}")
        
    def get_file(self): 
        """Get next file from the list
        
        Returns:
            str: File path or None if no more files
        """
        if self.files:
            return self.files.pop()
        return None
    
if __name__ == '__main__':
    f_base = FileBase('/Users/ranelkarimov/Library/Mobile Documents/com~apple~CloudDocs/Studium')
    print(f_base.traverse())