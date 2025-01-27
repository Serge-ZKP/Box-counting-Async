extern crate image;
use futures::future;
use image::{GenericImageView, ImageBuffer, Luma};
use std::collections::HashSet;

#[tokio::main]
async fn main() {
    // Load the image from a file path
    let img_path = "/home/titor/Downloads/carpet.png";
    let gray_img_path = "/home/titor/Downloads/carpet_gray.png";
    let binarized_img_path = "/home/titor/Downloads/carpet_binarized_image.png";

    let img = image::open(img_path).expect("Failed to open image");

    // Convert the image to grayscale
    let gray_img = img.to_luma8();
    gray_img
        .save(gray_img_path)
        .expect("Failed to save grayscale image");

    println!("Image has been converted to grayscale and saved.");

    // Define the threshold value
    let threshold = 128u8;

    // Binarize the grayscale image
    let binarized_img = binarize_image(&gray_img, threshold);
    binarized_img
        .save(binarized_img_path)
        .expect("Failed to save binarized image");

    println!("Image has been binarized and saved.");

    // Compute the box-counting dimensions asynchronously
    let (white_dim, black_dim) = tokio::join!(
        compute_box_counting_dimension_async(&binarized_img, true),
        compute_box_counting_dimension_async(&binarized_img, false)
    );

    println!("Box-counting dimension for white pixels: {}", white_dim);
    println!("Box-counting dimension for black pixels: {}", black_dim);
}

// Function to binarize a grayscale image
fn binarize_image(
    gray_img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    threshold: u8,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (width, height) = gray_img.dimensions();
    let mut binarized_img = ImageBuffer::new(width, height);

    for (x, y, pixel) in gray_img.enumerate_pixels() {
        let Luma([luma]) = *pixel;
        let binarized_pixel = if luma >= threshold { 255 } else { 0 };
        binarized_img.put_pixel(x, y, Luma([binarized_pixel]));
    }

    binarized_img
}

// Async function to compute the box-counting dimension
async fn compute_box_counting_dimension_async(
    binarized_img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    count_white: bool,
) -> f64 {
    let (width, height) = binarized_img.dimensions();
    let box_sizes = [2, 4, 8, 16, 32, 64, 128];

    // Process each box size asynchronously
    let box_counts = future::join_all(
        box_sizes
            .iter()
            .map(|&box_size| {
                let binarized_img = binarized_img.clone();
                async move {
                    let mut boxes = HashSet::new();

                    for y in (0..height).step_by(box_size as usize) {
                        for x in (0..width).step_by(box_size as usize) {
                            let mut found = false;

                            for dy in 0..box_size {
                                for dx in 0..box_size {
                                    if x + dx < width && y + dy < height {
                                        let pixel = binarized_img.get_pixel(x + dx, y + dy)[0];
                                        if (count_white && pixel == 255)
                                            || (!count_white && pixel == 0)
                                        {
                                            found = true;
                                            break;
                                        }
                                    }
                                }
                                if found {
                                    break;
                                }
                            }

                            if found {
                                boxes.insert((x / box_size, y / box_size));
                            }
                        }
                    }

                    (box_size, boxes.len() as f64)
                }
            })
            .collect::<Vec<_>>(),
    )
    .await;

    // Perform a linear fit to log-log data to estimate the fractal dimension
    let log_box_sizes: Vec<f64> = box_counts.iter().map(|&(s, _)| (s as f64).ln()).collect();
    let log_box_counts: Vec<f64> = box_counts.iter().map(|&(_, c)| c.ln()).collect();

    let (slope, _) = linear_regression(&log_box_sizes, &log_box_counts);

    // The fractal dimension is the negative of the slope
    -slope
}

// Helper function to perform linear regression
fn linear_regression(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let sum_x = xs.iter().sum::<f64>();
    let sum_y = ys.iter().sum::<f64>();
    let sum_xx = xs.iter().map(|&x| x * x).sum::<f64>();
    let sum_xy = xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum::<f64>();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    (slope, intercept)
}

