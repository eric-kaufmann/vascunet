import os
import meshio

class VesselIterator:
    def __init__(self, directory):
        self.vessel_files = self._get_vessel_files(directory)
        self.index = 0

    def __iter__(self):
        return self
    
    def __getitem__(self, index):
        vessel_file = self.vessel_files[index]
        return self._file_to_datadict(vessel_file)
    
    def __len__(self):
        return len(self.vessel_files)
    
    def __next__(self):
        if self.index < len(self.vessel_files):
            vessel_file = self.vessel_files[self.index]
            self.index += 1
            return self._file_to_datadict(vessel_file)
        else:
            raise StopIteration
    
    def _get_vessel_files(self, directory):
        vessel_files = []
        for filename in os.listdir(directory):
            if filename.endswith(".vtu"):
                file_path = os.path.join(directory, filename)
                vessel_files.append(file_path)
        return vessel_files
    
    def _file_to_datadict(self, filename):
        data = meshio.read(filename)
        data_dict = data.point_data
        data_dict["points"] = data.points
        data_dict["filename"] = filename
        return data_dict