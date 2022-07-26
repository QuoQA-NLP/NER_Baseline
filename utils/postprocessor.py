

class Postprocessor :

    def __init__(self, ) :
        self.mapping = {
            0:  0,  # B-DT
            1:  2,  # B-LC
            2:  4,  # B-OG
            3:  6,  # B-PS
            4:  8,  # B-QT
            5:  10, # B-TI
            6:  12, # O
        }

    def recover(self, pred_id) :
        pred_id = [self.mapping[p] for p in pred_id]

        result = [pred_id[0]]
        for i in range(1, len(pred_id)) :

            prev = pred_id[i-1]
            cur = pred_id[i]

            # B-DT, B-DT -> B-DT, I-DT
            if (prev == 0 or prev == 1) and cur == 0 :
                cur = 1

            # B-LC, B-LC -> B-LC, I-LC
            if (prev == 2 or prev == 3) and cur == 2 :
                cur = 3

            # B-OG, B-OG -> B-OG I-OG
            if (prev == 4 or prev == 5) and cur == 4 :
                cur = 5

            # B-PS, B-PS -> B-PS I-PS
            if (prev == 6 or prev == 7) and cur == 6 :
                cur = 7

            # B-QT, B-QT -> B-QT I-QT
            if (prev == 8 or prev == 9) and cur == 8 :
                cur = 9

            # B-TI, B-TI -> B-TI I-TI
            if (prev == 10 or prev == 11) and cur == 10 :
                cur = 11

            result.append(cur)

        return result