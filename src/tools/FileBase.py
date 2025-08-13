import os
import logging

logger = logging.getLogger('FileBase')

class FileBase():
    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.files: list[str] = None 

    def traverse(self) -> int:
        """_summary_

        Returns:
            int: number of files in directory
        """
        try: 
            stack = [self.dir]
            result = []

            while stack:
                curr = stack.pop()
                result.append(curr)
                if os.path.isdir(curr):
                    try:
                        children = [os.path.join(curr, child) for child in os.listdir(curr)]
                        stack.extend(reversed(children))
                    except PermissionError:
                        continue  
            result = list(filter(lambda x: "__" not in x or ".cpython" not in x,  result))
            self.files = result 
            logger.info(f'Traversed filebase. Found {len(result)} files')
            return len(result)
        except Exception as e: 
            logger.error(f"Failed to traverse FileBase: {e}")
    
    def file_content(self, file: str)-> str:
        try:    
            content: str 
            with open(file) as f: 
                content = f.read() 
            return content 

        except Exception as e: 
            logger.error(f"Failed to load File Content: {e}")
    
    def update_file_content(self, file: str, updated_content: str):
        try:  
            f_name = file.split('/')[-1]
            with open(file, "w") as f: 
                f.write(updated_content)
                logger.info(f'Updated file: {f_name}')
        except Exception as e: 
            logger.error(f"Failed to update File Content: {e}")
        
    def get_file(self): 
        return self.files.pop()
    
if __name__ == '__main__':
    f_base = FileBase('/Users/ranelkarimov/Library/Mobile Documents/com~apple~CloudDocs/Studium ')
    print(f_base.traverse())