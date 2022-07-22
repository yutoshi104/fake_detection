import subprocess
import gdown

url_list = [
    # ("https://drive.google.com/file/d/1biWqToAlee74V0TEKHxq65VdiJw8kSFF/view?usp=sharing","/data/toshikawa/DeeperForensics/lists.zip"),
    # ("https://drive.google.com/file/d/1AipCWdIXHpsxgZauNTZGW5S0Y9Uwfi4Q/view?usp=sharing","/data/toshikawa/DeeperForensics/manipulated_videos_part_00.zip"),
    # ("https://drive.google.com/file/d/1DYglmsAezQV1O7iQ3uO6bmQNKqMdUR3J/view?usp=sharing","/data/toshikawa/DeeperForensics/manipulated_videos_part_01.zip"),
    # ("https://drive.google.com/file/d/1e0aLZ66g5jZYNFojb141ENQ2qMfc_5xX/view?usp=sharing","/data/toshikawa/DeeperForensics/manipulated_videos_part_02.zip"),
    # ("https://drive.google.com/file/d/1T40QU_gP-uIA-ieljq_FNxABtPIddpQs/view?usp=sharing","/data/toshikawa/DeeperForensics/manipulated_videos_part_03.zip"),
    # ("https://drive.google.com/file/d/1xpK8SyNZpR1pRAB90dC_tHNF4LW8ondO/view?usp=sharing","/data/toshikawa/DeeperForensics/manipulated_videos_part_04.zip"),
    # ("https://drive.google.com/file/d/1iIEhjoRgOXMiT2OzlvUaMArr2LfV1pO5/view?usp=sharing","/data/toshikawa/DeeperForensics/manipulated_videos_part_05.zip"),
    # ("https://drive.google.com/file/d/1JoV8rm9CPfkhpcyFn2nPKmyIJX5uo_8Y/view?usp=sharing","/data/toshikawa/DeeperForensics/manipulated_videos_part_06.zip"),
    # ("https://drive.google.com/file/d/16NanHrF2MkVLFB6Huig4gBUU-j7xpFGH/view?usp=sharing","/data/toshikawa/DeeperForensics/manipulated_videos_part_07.zip"),
    # ("https://drive.google.com/file/d/1R4j946A3MyMlwn6kWR7W6obP7m9jhCD6/view?usp=sharing","/data/toshikawa/DeeperForensics/manipulated_videos_part_08.zip"),
    # ("https://drive.google.com/file/d/11mL_vr0RBkffnJLaGfvD9NIV5-Azen3E/view?usp=sharing","/data/toshikawa/DeeperForensics/manipulated_videos_part_09.zip"),
    # ("https://drive.google.com/file/d/1Gz4_0YZtW2ilot6HPRctKOINWxYf1B1S/view?usp=sharing","/data/toshikawa/DeeperForensics/manipulated_videos_part_10.zip"),
    # ("https://drive.google.com/file/d/1WxaC2aqMKsm4O7mJlacT8dgpPzFqkw6k/view?usp=sharing","/data/toshikawa/DeeperForensics/manipulated_videos_part_11.zip"),
    # ("https://drive.google.com/file/d/14PkJGixARVP_-fx5KfZkXnYUaWjXqXaR/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_00.zip"),
    # ("https://drive.google.com/file/d/1DXsOBHQeL3J2qrDK7tKYM8b3qy8mY0U3/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_01.zip"),
    # ("https://drive.google.com/file/d/1MKWs5zjgRqGi0Pm7OFcWnRidsvmd3UdE/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_02.zip"),
    # ("https://drive.google.com/file/d/1CbgzfJW1kZjJJulAbPOhBN5ZpbFzNLLA/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_03.zip"),
    # ("https://drive.google.com/file/d/1qkfCsrOVx5K1lHI202-qvdB7MohHpaH8/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_04.zip"),
    # ("https://drive.google.com/file/d/13jBHfUp-zdfccLHkeZP32-xJ6gEvPFW4/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_05.zip"),
    # ("https://drive.google.com/file/d/1bxsmBtBxE_XHf4S8Xb38mT657ZsE52kK/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_06.zip"),
    # ("https://drive.google.com/file/d/1t5yr4boCJJTcuvrLzwSImd6_IfY2wFHB/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_07.zip"),
    # ("https://drive.google.com/file/d/1OKnyU48JL5vxehoLXrChFYQr0WHPtFWm/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_08.zip"),
    # ("https://drive.google.com/file/d/1dj3tkzohngGHyJiZxJ2rs6HjwvvqA0BZ/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_09.zip"),
    # ("https://drive.google.com/file/d/1PuX99DJSRn40hf6j9e1-3RmYnhFL-0Kr/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_10.zip"),
    # ("https://drive.google.com/file/d/17NtM2lqK0JFjPPk7LSkdS_r8IPTe7DW3/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_11.zip"),
    # ("https://drive.google.com/file/d/15Nso97nVJYyruGYMCzESRbQGKrQo3M32/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_12.zip"),
    # ("https://drive.google.com/file/d/1OaLToPDd7IL0bg4NsT0sd3oMdTdU978Z/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_13.zip"),
    # ("https://drive.google.com/file/d/1SAog6n70oDIw3o5i0oUsrpz65MfrB4T-/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_14.zip"),
    # ("https://drive.google.com/file/d/1xkoH8lyGVNAW3aVVvTWuXBoEdJvNl_9I/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_15.zip"),
    # ("https://drive.google.com/file/d/1hbK2rMO5Ku0KGFTDf7lPgZ0imICnElm8/view?usp=sharing","/data/toshikawa/DeeperForensics/source_videos_part_16.zip"),
]


for url,name in url_list:
    # try:
    #     cmd = f'wget "{url}" -O {name}'
    #     print(cmd)
    #     res = subprocess.check_output(cmd)
    # except:
    #     print(f"Error: {name}")
    # else:
    #     print(f"Result of {name}: "+res)

    id = url[32:65]
    print(id)
    new_url = f"https://drive.google.com/uc?id={id}"
    gdown.download(new_url,name,quiet=False)
