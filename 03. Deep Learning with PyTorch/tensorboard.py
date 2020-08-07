# Import the libraries
import math
from tensorboardX import SummaryWriter

#
if __name__ == "__main__":
    
    # Writer of data
    writer = SummaryWriter()

    # Functions that are going to be visualized
    funcs = {"sin": math.sin, "cos": math.cos, "tan": math.tan}

    # Loop over angle ranges in degrees
    for angle in range(-360, 360):
        
        # Convert the angle range (in degrees) into radians
        angle_rad = angle * math.pi / 180
        
        # Loop over functions and their names
        for name, fun in funcs.items():
            
            # Calculate our functions' values
            val = fun(angle_rad)
            
            # Add every value to the writer
            writer.add_scalar(name, val, angle)

    # Close the writer
    writer.close()