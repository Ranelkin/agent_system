import os
from ..shared.log_config import setup_logging

logger = setup_logging('FileBase')

class FileBase():
    def __init__(self, dir: str) -> None:
        """
        Initializes the FileBase object with a directory path.
        
        Args:
            dir (str): Path to the directory to traverse.
        """
        self.dir = dir
        logger.info(f"File Base initialized with dir: {dir}")
        self.files: list[str] = None
        self.num_files = None 
        
    def traverse(self) -> None:
        """
        Traverses the directory tree starting from self.dir, collecting all file paths.
        Filters out files with names containing '__' or '.cpython'.

        Returns:
            int: Number of collected files.
        """
        try:
            stack = [self.dir]
            result = []

            while stack:
                curr = stack.pop()
                if os.path.isdir(curr):
                    try:
                        children = [os.path.join(curr, child) for child in os.listdir(curr)]
                        stack.extend(reversed(children))
                    except PermissionError:
                        continue
                else:
                    result.append(curr)  

            # Filter out files with specific substrings
            result = [f for f in result if f.endswith('.py') and "__" not in f and ".cpython" not in f]
            self.files = result
            logger.info(f'Traversed filebase. Found {len(result)} files')
            self.num_files = len(result)
        except Exception as e:
            logger.error(f"Failed to traverse FileBase: {e}")

    def file_content(self, file: str) -> str:
        """
        Reads and returns the content of a specified file.

        Args:
            file (str): Path to the file to read.

        Returns:
            str: Content of the file.
        """
        try:
            content: str
            with open(file, 'r') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Failed to load file content from {file}: {e}")

    def update_file_content(self, file: str, updated_content: str):
        """
        Overwrites the content of the specified file with provided content.

        Args:
            file (str): Path to the file to update.
            updated_content (str): New content to write into the file.
        """
        def write_file(f): 
            with open(f, 'w') as f:
                f.write(updated_content)
            logger.info(f'Wrote file: {f_name}')
        try:
            f_name = os.path.basename(file)
            if os.path.exists(file):
                write_file(file)
            else: 
                os.makedirs(file, exist_ok=True)
                write_file(file)
                
        except Exception as e:
            logger.error(f"Failed to update file content for {file}: {e}")

    def get_file(self) -> str:
        """
        Retrieves and removes the last file from the list of files.

        Returns:
            str: Path of the file popped from the list.
        """
        if self.files:
            return self.files.pop()
        return None

