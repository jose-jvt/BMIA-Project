import subprocess
import os

class DSIStudioPipeline:
    def __init__(self, dsi_studio_path, input_path, output_prefix, atlas="0", bval_file=None, bvec_file=None):
        self.dsi_studio_path = dsi_studio_path
        self.input_path = input_path
        self.output_prefix = output_prefix
        self.atlas = atlas
        self.bval_file = bval_file
        self.bvec_file = bvec_file
        self.input_type = self.__detect_input_type()

    def __detect_input_type(self):
        if os.path.isdir(self.input_path):
            files = os.listdir(self.input_path)
            if any(f.lower().endswith(".dcm") for f in files):
                return "dicom"
            elif any(f.lower().endswith(".fdf") or "2dseq" in f.lower() for f in files):
                return "varian"
        elif self.input_path.lower().endswith(".nii") or self.input_path.lower().endswith(".nii.gz"):
            return "nifti"
        raise ValueError("Unrecognized input format. Provide .nii, DICOM folder, or Varian folder.")

    def __create_src(self):
        print(f"[Step 1] Creating .sz from {self.input_type} input...")
        cmd = [self.dsi_studio_path, "--action=src"]
        if self.input_type == "nifti":
            if not self.bval_file or not self.bvec_file:
                raise FileNotFoundError("Missing .bval and/or .bvec for NIfTI input.")
            cmd += [
                f"--source={self.input_path}",
                f"--bval={self.bval_file}",
                f"--bvec={self.bvec_file}"
            ]
        else:
            cmd += [f"--source={self.input_path}"]
        cmd += [f"--output={self.output_prefix}.sz"]
        subprocess.run(cmd, check=True)

    def __reconstruct_dti(self):
        print("[Step 2] Reconstructing DTI...")
        cmd = [
            self.dsi_studio_path,
            "--action=rec",
            f"--source={self.output_prefix}.sz",
            "--method=4",
            "--param0=1.25",
            f"--output={self.output_prefix}.dti.fib.gz"
        ]
        subprocess.run(cmd, check=True)

    def __run_tracking(self):
        print("[Step 3] Running tractography...")
        cmd = [
            self.dsi_studio_path,
            "--action=trk",
            f"--source={self.output_prefix}.dti.fib.gz",
            f"--output={self.output_prefix}.tt.gz"
        ]
        subprocess.run(cmd, check=True)

    def __run_connectivity_analysis(self):
        print("[Step 4] Generating connectivity matrix...")
        cmd = [
            self.dsi_studio_path,
            "--action=ana",
            f"--source={self.output_prefix}.dti.fib.gz",
            f"--tract={self.output_prefix}.tt.gz",
            f"--connectivity={self.atlas}",
            f"--output={self.output_prefix}"
        ]
        subprocess.run(cmd, check=True)

    def run(self):
        try:
            self.__create_src()
            self.__reconstruct_dti()
            self.__run_tracking()
            self.__run_connectivity_analysis()
            print("Pipeline complete. Connectivity matrix saved as connectivity.mat.")
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed: {e}")
        except Exception as e:
            print(f"Error: {e}")

# === USAGE EXAMPLE ===
if __name__ == "__main__":
    pipeline = DSIStudioPipeline(
        dsi_studio_path=r"C:\Users\pall2\Desktop\BMIA\dsi_studio_win\dsi_studio.exe",
        input_path=r"C:\Users\pall2\Downloads\sub-01_dwi.nii.gz",  # Can be .nii.gz, DICOM folder, or Varian folder
        output_prefix=r"C:\Users\pall2\Downloads\sub-01_dwi_out",  # Output prefix for all files
        atlas="Brodmann",  # AAL
        bval_file=r"C:\Users\pall2\Downloads\sub-01_dwi.bval",  # Only for NIfTI
        bvec_file=r"C:\Users\pall2\Downloads\sub-01_dwi.bvec"   # Only for NIfTI
    )
    pipeline.run()
