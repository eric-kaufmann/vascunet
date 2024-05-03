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
        for fluid_filename in os.listdir(directory):
            if fluid_filename.endswith('fluid.vtu'):
                fluid_file_path = os.path.join(directory, fluid_filename)
                mesh_file_path = os.path.join(directory, fluid_file_path.replace('fluid.vtu', 'wss.vtu'))
                vessel_files.append({'fluid_file': fluid_file_path, 'mesh_file': mesh_file_path})
        return vessel_files
    
    def _file_to_datadict(self, filename):
        
        fluid_data = meshio.read(filename['fluid_file'])
        mesh_data = meshio.read(filename['mesh_file'])

        data_dict = fluid_data.point_data
        data_dict['fluid_file_path'] = filename['fluid_file']
        data_dict['mesh_file_path'] = filename['mesh_file']
        data_dict['data_points'] = fluid_data.points
        data_dict['mesh_points'] = mesh_data.points
        return data_dict
