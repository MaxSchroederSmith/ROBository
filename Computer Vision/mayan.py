def main(debug=False):

        
    if lights:
        allOn(strip)
    init_camera()
    capture_dir = output_dir+'/pi_card.jpg'

    found = False

    while not found
        camera.capture(capture_dir)
        img = cv2.imread(capture_dir)
        
        img = crop_image(img)  # MAYBE WANT TO VALIDATE THIS AS WELL?
        save_image("1cropped",img)
        valid, extracted_rank = extract_rank(img, "test.jpg")
        # todo Save the extracted image into the card_imgs file
        # DOESN'T KNOW WHAT THE SUIT IS YET THO????
        # FUNCTION LATER ON SORTING INTO CORRECT FOLDER??

        train_ranks = load_ranks(rank_dataset_dir)
        best_rank_match, rank_diff = match(extracted_rank, train_ranks)
        if (best_rank_match != "Unknown"):
            found = True

    camera.close()
    if lights:
        allOff(strip)
    
    return best_rank_match
