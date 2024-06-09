from tqdm import tqdm

def loading_bar(iter, text, disable_logs = True):
    '''Pass in an iterable to display a loading bar duing a for loop'''
    
    return tqdm(iter, bar_format=f'{text:<30}\t {'{percentage:3.0f}'}% |{'{bar}'}|', ncols=100, leave=False, disable=disable_logs)

def println(text: str, disable_logs: bool = True) -> None:
    '''Prints a line that will be erased later'''
    
    if disable_logs: return
    print(f'{text:<150}', end='\r')