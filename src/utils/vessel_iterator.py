import os
import meshio
import numpy as np

class VesselIterator:
    """
    Iterator class for iterating over vessel files in a directory.
    """

    def __init__(self, directory):
        """
        Initializes a VesselIterator object.

        Parameters:
        - directory (str): The directory containing the vessel files.
        """
        self.vessel_files = self._get_vessel_files(directory)
        self.index = 0

    def __iter__(self):
        """
        Returns the iterator object itself.
        """
        return self
    
    def __getitem__(self, index):
        """
        Returns the data dictionary for the vessel file at the given index.

        Parameters:
        - index (int): The index of the vessel file.

        Returns:
        - dict: The data dictionary for the vessel file.
        """
        vessel_file = self.vessel_files[index]
        return self._file_to_datadict(vessel_file)
    
    def __len__(self):
        """
        Returns the number of vessel files in the iterator.

        Returns:
        - int: The number of vessel files.
        """
        return len(self.vessel_files)
    
    def __next__(self):
        """
        Returns the next data dictionary for the vessel file in the iterator.

        Returns:
        - dict: The data dictionary for the next vessel file.

        Raises:
        - StopIteration: If there are no more vessel files in the iterator.
        """
        if self.index < len(self.vessel_files):
            vessel_file = self.vessel_files[self.index]
            self.index += 1
            return self._file_to_datadict(vessel_file)
        else:
            raise StopIteration
    
    def _get_vessel_files(self, directory):
        """
        Retrieves the vessel files from the given directory.

        Parameters:
        - directory (str): The directory containing the vessel files.

        Returns:
        - list: A list of dictionaries, where each dictionary contains the paths to the fluid and mesh files.
        """
        vessel_files = []
        for fluid_filename in os.listdir(directory):
            if fluid_filename.endswith('fluid.vtu'):
                fluid_file_path = os.path.join(directory, fluid_filename)
                mesh_file_path = os.path.join(directory, fluid_file_path.replace('fluid.vtu', 'wss.vtu'))
                vessel_files.append({'fluid_file': fluid_file_path, 'mesh_file': mesh_file_path})
        return vessel_files
    
    def _file_to_datadict(self, filename):
        """
        Converts a vessel file to a data dictionary.

        Parameters:
        - filename (dict): A dictionary containing the paths to the fluid and mesh files.

        Returns:
        - dict: The data dictionary for the vessel file.
        """
        fluid_data = meshio.read(filename['fluid_file'])
        mesh_data = meshio.read(filename['mesh_file'])

        data_dict = fluid_data.point_data
        data_dict['fluid_file_path'] = filename['fluid_file']
        data_dict['mesh_file_path'] = filename['mesh_file']
        data_dict['data_points'] = fluid_data.points
        data_dict['mesh_points'] = mesh_data.points
        return data_dict


def main():
    mesh_data_points = []
    fluid_data_points = []
    velocity_systolic_data = []
    velocity_diastolic_data = []
    pressure_systolic_data = []
    pressure_diastolic_data = []

    directory = r"C:\Users\Eric Kaufmann\workspace\MA\data\carotid_flow_database"
    for fluid_filename in os.listdir(directory):
        if fluid_filename.endswith('fluid.vtu'):
            fluid_file_path = os.path.join(directory, fluid_filename)
            mesh_file_path = os.path.join(directory, fluid_file_path.replace('fluid.vtu', 'wss.vtu'))

            fluid_data = meshio.read(fluid_file_path)
            mesh_data = meshio.read(mesh_file_path)


            mesh_data_points.append(mesh_data.points)
            fluid_data_points.append(fluid_data.points)
            velocity_systolic_data.append(fluid_data.point_data['velocity_systolic'])
            velocity_diastolic_data.append(fluid_data.point_data['velocity_diastolic'])
            pressure_systolic_data.append(fluid_data.point_data['pressure_systolic'])
            pressure_diastolic_data.append(fluid_data.point_data['pressure_diastolic'])

    mesh_data_points = np.array(mesh_data_points)
    fluid_data_points = np.array(fluid_data_points)
    velocity_systolic_data = np.array(velocity_systolic_data)
    velocity_diastolic_data = np.array(velocity_diastolic_data)
    pressure_systolic_data = np.array(pressure_systolic_data)
    pressure_diastolic_data = np.array(pressure_diastolic_data)
    
    print(f"Mesh data points: {mesh_data_points.shape}")
    print(mesh_data_points[0])
    print()
    print(f"Fluid data points: {fluid_data_points.shape}")
    print(fluid_data_points[0])
    print()
    print(f"Velocity systolic data: {velocity_systolic_data.shape}")
    print(velocity_systolic_data[0])
    print()
    print(f"Velocity diastolic data: {velocity_diastolic_data.shape}")
    print(velocity_diastolic_data[0])
    print()
    print(f"Pressure systolic data: {pressure_systolic_data.shape}")
    print(pressure_systolic_data[0])
    print()
    print(f"Pressure diastolic data: {pressure_diastolic_data.shape}")
    print(pressure_diastolic_data[0])
    print()



if __name__ == '__main__':
    main()
    # DATA_DIR = r"C:\Users\Eric Kaufmann\workspace\MA\data\carotid_flow_database"
    # vessels = VesselIterator(DATA_DIR)
    # mesh_vessel_points = []
    # fluid_vessel_points = []
    # for v_idx, vessel in enumerate(vessels):
    #     # if v_idx == 10:
    #     #     break
    #     print(f"Vessel {v_idx+1}/{len(vessels)}:")
    #     print(f"# vessel mesh points: {len(vessel['mesh_points'])}")
    #     print(f"# vessel fluid points: {len(vessel['data_points'])}")
    #     print()
    #     mesh_vessel_points.append(len(vessel['mesh_points']))
    #     fluid_vessel_points.append(len(vessel['data_points']))
    
    # print(f"Minimum mesh points: {min(mesh_vessel_points)}")
    # print(f"Maximum mesh points: {max(mesh_vessel_points)}")
    # print(f"Average mesh points: {sum(mesh_vessel_points)/len(mesh_vessel_points)}")

    # print(f"Minimum fluid points: {min(fluid_vessel_points)}")
    # print(f"Maximum fluid points: {max(fluid_vessel_points)}")
    # print(f"Average fluid points: {sum(fluid_vessel_points)/len(fluid_vessel_points)}")

