use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

pub type Fvec = DataVec<f32>;
pub type Ivec = DataVec<u32>;

pub struct DataVec<T> {
    pub data: Vec<T>,
    pub dim: usize,
    pub num: usize,
}

pub enum CommonVec {
    Fvec,
    Ivec,
}



impl<T> DataVec<T> {
    pub fn new(dim: usize, num: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), dim * num);
        Self { data, dim, num }
    }
    pub fn merge(&mut self, other: Self) {
        assert_eq!(self.dim, other.dim);
        self.data.extend(other.data);
        self.num += other.num;
    }
    /// return (dim, num)
    pub fn read_size(file_path: &Path) -> (usize, usize) {
        let mut file = File::open(file_path).unwrap();
        let mut k = [0u8; 4];
        file.read_exact(&mut k).unwrap();
        let dim: u32 = u32::from_le_bytes(k);
        file.seek(SeekFrom::End(0)).unwrap();
        let fsize = file.stream_position().unwrap() as usize;
        let num = fsize / ((dim as usize + 1) * 4);
        (dim as usize, num)
    }

    pub fn get_node(&self, node_id: usize) -> &[T] {
        &self.data[node_id * self.dim..(node_id + 1) * self.dim]
    }
}
pub trait ScalaData {
    fn zero() -> Self;
    fn from_le_bytes(bytes: [u8; 4]) -> Self;
    fn to_le_bytes(self) -> [u8; 4];
}
impl ScalaData for f32 {
    fn zero() -> Self {
        0.0
    }
    fn from_le_bytes(bytes: [u8; 4]) -> Self {
        f32::from_le_bytes(bytes)
    }
    fn to_le_bytes(self) -> [u8; 4] {
        self.to_le_bytes()
    }
}
impl ScalaData for u32 {
    fn zero() -> Self {
        0
    }
    fn from_le_bytes(bytes: [u8; 4]) -> Self {
        u32::from_le_bytes(bytes)
    }
    fn to_le_bytes(self) -> [u8; 4] {
        self.to_le_bytes()
    }
}

impl<T: ScalaData + Clone + Copy> DataVec<T> {
    // Helper function to read data from file
    fn read_data_from_file(file: &mut File, dim: u32, start: usize, end: usize) -> Vec<T> {
        let mut data: Vec<T> = Vec::new();
        for _row_id in start..end {
            file.seek(SeekFrom::Current(4)).unwrap();
            let mut bytes = vec![0u8; dim as usize * 4];
            file.read_exact(&mut bytes).unwrap();
            let mut buffer = vec![T::zero(); dim as usize];

            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                buffer[i] = T::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                // check the correctness of the fvec file
            }

            data.extend(buffer);
        }
        data
    }
    pub fn from_file_slice(file_path: &Path, start: usize, end: usize) -> Self {
        let mut data: Vec<T> = Vec::new();
        let mut file = File::open(file_path).unwrap();
        let mut k = [0u8; 4];
        file.read_exact(&mut k).unwrap();
        let dim: u32 = u32::from_le_bytes(k);
        file.seek(SeekFrom::End(0)).unwrap();
        let fsize = file.stream_position().unwrap() as usize;
        let num = fsize / ((dim as usize + 1) * 4);

        assert!(end <= num);
        assert!(start < end);
        file.seek(SeekFrom::Start((start * (dim as usize + 1) * 4) as u64))
            .unwrap();
        let partial_num = end - start;
        data.reserve(partial_num * dim as usize);

        data.extend(Self::read_data_from_file(&mut file, dim, start, end));

        Self {
            data,
            dim: dim as usize,
            num: partial_num,
        }
    }
    pub fn from_file(file_path: &Path) -> Self {
        let mut data = Vec::new();
        let mut file = File::open(file_path).unwrap();
        let mut k: [u8; 4] = [0u8; 4];
        file.read_exact(&mut k).unwrap();
        let dim: u32 = u32::from_le_bytes(k);
        file.seek(SeekFrom::End(0)).unwrap();
        let fsize = file.stream_position().unwrap() as usize;
        let num = fsize / ((dim as usize + 1) * 4);
        data.reserve(num * dim as usize);
        file.seek(SeekFrom::Start(0)).unwrap();

        data.extend(Self::read_data_from_file(&mut file, dim, 0, num));

        Self {
            data,
            dim: dim as usize,
            num,
        }
    }
    pub fn save(&self, file_path: &Path) {
        let mut file = File::create(file_path).unwrap();
        for i in 0..self.num {
            file.write_all(&(self.dim as u32).to_le_bytes()).unwrap();
            let mut buffer = vec![0u8; self.dim * 4];
            let data = self.get_node(i);
            for j in 0..self.dim {
                let bytes = data[j].to_le_bytes();
                buffer[j * 4..(j + 1) * 4].copy_from_slice(&bytes);
            }
            file.write_all(&buffer).unwrap();
        }
    }
}

impl DataVec<f32> {
    pub fn get_center_point(&self) -> Vec<f32> {
        let mut center = vec![f32::default(); self.dim];
        for i in 0..self.num {
            for j in 0..self.dim {
                center[j] += self.data[i * self.dim + j];
            }
        }
        for j in 0..self.dim {
            center[j] /= self.num as f32;
        }
        center
    }
}

impl<T: Clone> DataVec<T> {
    // split and return a new Fvec of 0..size
    pub fn split(&self, size: usize) -> Self {
        assert!(size < self.num);
        Self {
            data: self.data[0..size * self.dim].to_vec(),
            dim: self.dim,
            num: size,
        }
    }

    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(start < end);
        assert!(end <= self.num);
        Self {
            data: self.data[start * self.dim..end * self.dim].to_vec(),
            dim: self.dim,
            num: end - start,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore]
    fn test_read_fvec() {
        let fvec = super::Fvec::from_file(std::path::Path::new(
            "/home/sjq/sjqssd/nsg-fork/gist_base.fvecs",
        ));
        assert_eq!(fvec.dim, 960);
        assert_eq!(fvec.num, 1000000);
        assert_eq!(fvec.data.len(), 1000000 * 960);
        for i in 0..10 {
            println!("{}", fvec.data[i])
        }
        for i in ((1000000 - 1) * 960..1000000 * 960).take(10) {
            println!("{}", fvec.data[i])
        }
    }

    #[test]
    fn test_read_write() {
        let fvec = super::Fvec::new(2, 2, vec![1., 2., 3., 4.]);
        let uuid = uuid::Uuid::new_v4();
        let file_name = format!("test_{}.fvecs", uuid);
        fvec.save(std::path::Path::new(&file_name));
        let fvec2 = super::Fvec::from_file(std::path::Path::new(&file_name));
        assert_eq!(fvec.dim, fvec2.dim);
        assert_eq!(fvec.num, fvec2.num);
        assert_eq!(fvec.data.len(), fvec2.data.len());
        for i in 0..4 {
            assert_eq!(fvec.data[i], fvec2.data[i]);
        }
        // delete the file
        std::fs::remove_file(&file_name).unwrap();
    }

    #[test]
    fn test_split() {
        let fvec = super::Fvec::new(2, 2, vec![1., 2., 3., 4.]);
        let fvec2 = fvec.split(1);
        assert_eq!(fvec2.dim, 2);
        assert_eq!(fvec2.num, 1);
        assert_eq!(fvec2.data.len(), 2);
        for i in 0..2 {
            assert_eq!(fvec.data[i], fvec2.data[i]);
        }
    }

    #[test]
    fn test_slice() {
        let fvec = super::Fvec::new(2, 2, vec![1., 2., 3., 4.]);
        let fvec2 = fvec.slice(0, 1);
        assert_eq!(fvec2.dim, 2);
        assert_eq!(fvec2.num, 1);
        assert_eq!(fvec2.data.len(), 2);
        for i in 0..2 {
            assert_eq!(fvec.data[i], fvec2.data[i]);
        }
        let fvec3 = fvec.slice(1, 2);
        assert_eq!(fvec3.dim, 2);
        assert_eq!(fvec3.num, 1);
        assert_eq!(fvec3.data.len(), 2);
        for i in 0..2 {
            assert_eq!(fvec.data[2 + i], fvec3.data[i]);
        }
    }

    #[test]
    fn test_from_file_slice() {
        let fvec = super::Fvec::new(2, 2, vec![1., 2., 3., 4.]);
        let uuid = uuid::Uuid::new_v4();
        let file_name = format!("test_{}.fvecs", uuid);
        fvec.save(std::path::Path::new(&file_name));
        let fvec2 = super::Fvec::from_file_slice(std::path::Path::new(&file_name), 0, 1);
        assert_eq!(fvec2.dim, 2);
        assert_eq!(fvec2.num, 1);
        assert_eq!(fvec2.data.len(), 2);
        for i in 0..2 {
            assert_eq!(fvec.data[i], fvec2.data[i]);
        }
        let fvec3 = super::Fvec::from_file_slice(std::path::Path::new(&file_name), 1, 2);
        assert_eq!(fvec3.dim, 2);
        assert_eq!(fvec3.num, 1);
        assert_eq!(fvec3.data.len(), 2);
        for i in 0..2 {
            assert_eq!(fvec.data[2 + i], fvec3.data[i]);
        }
        // delete the file
        std::fs::remove_file(&file_name).unwrap();
    }

    #[test]
    fn test_read_size() {
        let file = super::Fvec::new(2, 2, vec![1., 2., 3., 4.]);
        let uuid = uuid::Uuid::new_v4();
        let file_name = format!("test_{}.fvecs", uuid);

        file.save(std::path::Path::new(&file_name));
        let (dim, num) = super::Fvec::read_size(std::path::Path::new(&file_name));
        assert_eq!(dim, 2);
        assert_eq!(num, 2);
        // delete the file
        std::fs::remove_file(&file_name).unwrap();
    }
}
