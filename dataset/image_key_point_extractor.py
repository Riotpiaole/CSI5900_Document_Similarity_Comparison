import cv2 as cv

def orb_feature(image, dim=True):
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    orb = cv.ORB_create()
    return orb.detectAndCompute(image, None)
    
def orb_feature_matcher(query , target , show=False):
    matcher = cv.BFMatcher()
    query_keypoints, query_descriptor = orb_feature(query)
    train_keypoints, target_descriptor = orb_feature(target)
    matches = matcher.match(query_descriptor, target_descriptor)
    if show:
        final_img = cv.drawMatches(
            query, query_keypoints, 
            target, train_keypoints, 
                matches[:20], None)
        
        final_img = cv.resize(final_img, (1000, 650))
        cv.imshow("Matches", final_img)
        k = cv.waitKey(0)
        if k == 27: 
            return 